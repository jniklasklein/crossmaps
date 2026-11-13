#!/usr/bin/env python3
"""Central two-rover mapping experiment. No motion or robot-TF publishers.

python3 crossmaps_dino_experimental/crossmaps_collaborative.py --demo
python3 crossmaps_dino_experimental/crossmaps_collaborative.py --config ...json
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

import cv2
import numpy as np

from collab_backend import Backend
from collab_config import ROOT, load_config, owned_run_directory
from collab_dashboard import CollaborativeDashboard
from collab_runtime import FairLatestScheduler, CycleTiming
from collab_transport import run_adapter
from pipeline_instance import PipelineInstance


class FleetApplication:
    """Owns all writes to backend state and exposes immutable UI snapshots."""
    def __init__(self, config, run_dir, *, demo=False, fleet=None, replay=False):
        self.config, self.run_dir, self.demo, self.replay = config, Path(run_dir), demo, replay
        if replay:
            if not (self.run_dir / 'evidence' / 'state.json').is_file():
                raise ValueError('Replay requires a completed collaborative evidence checkpoint')
            self.backend = Backend.replay(self.run_dir / 'evidence')
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.backend = Backend(config.get('backend'), self.run_dir / 'evidence')
        self.backend_lock = threading.RLock()
        self.state_lock = threading.RLock()
        self.cached = self.backend.snapshot()
        self.scheduler = FairLatestScheduler([r['id'] for r in config['robots']], config['input']['max_age_s'])
        self.timing = {r['id']: CycleTiming() for r in config['robots']}
        self.total_timing = CycleTiming()
        self.telemetry = {r['id']: dict(id=r['id'], robot_id=r['id'], domain=r['domain'],
            status='waiting_for_input', frame_age_s=None, observation_age_s=None) for r in config['robots']}
        self.images = {}
        self.events = deque(maxlen=40)
        self.stop_event = threading.Event()
        self.context = None
        self.fleet = fleet
        self.worker = None
        self.adapters = {}
        self.mp_context = mp.get_context('spawn')
        self.input_stop = self.mp_context.Event()
        self.dashboard = CollaborativeDashboard(self.snapshot, self.command, self.image,
                                                **config['dashboard'])
        self.active_robot = None
        self.active_since = None
        self.publisher_node = self.state_pub = None
        self.started = time.monotonic()
        self.last_ros_publish = 0.
        self._closed = False
        if replay:
            for source in self.cached.get('robots', []):
                self.telemetry.setdefault(source['robot_id'], {}).update(
                    session_id=source['session_id'], status='recorded')
            for frame in self.backend.keyframes.values():
                robot = frame['robot_id']
                ok, image = cv2.imencode('.jpg', frame['rgb'])
                if ok:
                    self.images[(robot, 'rgb')] = (image.tobytes(), self.started)
                depth = frame['depth']
                color = cv2.applyColorMap(np.clip(255*(1-depth/8), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                color[~np.isfinite(depth) | (depth <= 0)] = 0
                ok, image = cv2.imencode('.jpg', color)
                if ok:
                    self.images[(robot, 'depth')] = (image.tobytes(), self.started)

    def event(self, kind, message):
        with self.state_lock:
            self.events.append(dict(time=time.time(), kind=kind, message=str(message)[:500]))

    def offer(self, observation):
        robot = observation['robot_id']
        if robot not in self.telemetry:
            raise ValueError('unknown source')
        if self.replay:
            raise ValueError('Recorded replay does not accept live input')
        if not self.scheduler.offer(observation):
            return False
        now = time.monotonic()
        # Preview is acquisition-driven, never blocked by semantic inference.
        bgr = observation['rgb']
        ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with self.state_lock:
            if ok:
                self.images[(robot, 'rgb')] = (jpeg.tobytes(), now)
            state = self.telemetry[robot]
            state.update(last_frame_monotonic=now, session_id=observation['session_id'],
                         last_input_stamp_ns=observation['stamp_ns'],
                         status='processing' if self.active_robot == robot else 'receiving',
                         depth_registration=observation.get('depth_registration'),
                         calibration_id=observation.get('calibration_id'))
        return True

    def process_one(self):
        observation = self.scheduler.take()
        if observation is None:
            return False
        robot = observation['robot_id']
        started = time.monotonic()
        with self.state_lock:
            self.active_robot, self.active_since = robot, started
            self.telemetry[robot].update(status='processing', last_error=None)
        try:
            if self.demo:
                enriched = observation
                inference_elapsed = 0.
            else:
                if self.fleet is None:
                    from collab_perception import PerceptionFleet
                    self.fleet = PerceptionFleet(self.config, context=self.context)
                enriched = self.fleet.process(observation)
                inference_elapsed = time.monotonic()-started
            with self.backend_lock:
                snapshot = self.backend.ingest(enriched)
                with self.state_lock:
                    self.cached = snapshot
            finished = time.monotonic()
            with self.state_lock:
                self.timing[robot].add(finished-started, inference_elapsed,
                                      finished-observation['received_monotonic'])
                self.total_timing.add(finished-started, inference_elapsed,
                                      finished-observation['received_monotonic'])
                allowed = enriched.get('integration_allowed', True)
                self.telemetry[robot].update(status='healthy' if allowed else 'pose_rejected',
                    last_processed_monotonic=finished,
                    last_processed_stamp_ns=observation['stamp_ns'], pose_quality=enriched.get('pose_quality'),
                    counts=dict(masks=enriched.get('mask_count'), segments=enriched.get('segment_count'),
                                entities=enriched.get('entity_count')))
                if self.demo:
                    for view, bgr in enriched.get('demo_images', {}).items():
                        ok, image = cv2.imencode('.jpg', bgr)
                        if ok:
                            self.images[(robot, view)] = (image.tobytes(), finished)
            return True
        except Exception as error:
            with self.state_lock:
                self.telemetry[robot].update(status='processing_error', last_error=str(error)[:500])
            self.event('error', f'{robot}: {type(error).__name__}: {error}')
            print(f'[collaboration] {robot}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
            return False
        finally:
            with self.state_lock:
                self.active_robot = self.active_since = None

    def _work(self):
        while not self.stop_event.is_set():
            if not self.process_one():
                self.stop_event.wait(.05)

    def _start_adapters(self):
        for robot in self.config['robots']:
            output, health = self.mp_context.Queue(maxsize=1), self.mp_context.Queue(maxsize=2)
            process = self.mp_context.Process(target=run_adapter,
                args=(robot, self.config['input'], output, health, self.input_stop),
                name=f"crossmaps-input-{robot['id']}", daemon=True)
            process.start()
            self.adapters[robot['id']] = dict(process=process, output=output, health=health)

    def start(self, live=True):
        self.dashboard.start()
        try:
            if live and not self.replay:
                import rclpy
                from rclpy.context import Context
                from rclpy.node import Node
                from std_msgs.msg import String
                self.context = Context()
                rclpy.init(args=[], context=self.context, domain_id=self.config['central_domain'])
                self.publisher_node = Node('crossmaps_collaborative_server', context=self.context)
                self.state_pub = self.publisher_node.create_publisher(String, '/crossmaps/collaboration/state', 1)
                self._start_adapters()
            if not self.replay:
                self.worker = threading.Thread(target=self._work, name='shared-perception', daemon=True)
                self.worker.start()
            return self
        except BaseException:
            self.close()
            raise

    def poll_inputs(self):
        for robot, adapter in self.adapters.items():
            if not adapter['process'].is_alive():
                with self.state_lock:
                    self.telemetry[robot].update(status='adapter_exited', last_error=f"Input adapter exited ({adapter['process'].exitcode}); restart server after inspecting terminal")
                continue
            for _ in range(4):
                try:
                    health = adapter['health'].get_nowait()
                    preview = health.pop('preview_jpeg', None)
                    with self.state_lock:
                        self.telemetry[robot]['health'] = health
                        if preview:
                            self.images[(robot, 'rgb')] = (preview, float(health.get('preview_monotonic', time.monotonic())))
                        if health.get('last_error'):
                            self.telemetry[robot].update(status='input_rejected', last_error=health['last_error'])
                except queue.Empty:
                    break
            try:
                self.offer(adapter['output'].get_nowait())
            except queue.Empty:
                pass
        if self.publisher_node and time.monotonic()-self.last_ros_publish > 1.:
            from std_msgs.msg import String
            snapshot = self.snapshot()
            # ROS diagnostics carry metadata, never a second dense-map stream.
            snapshot.pop('submaps', None)
            for entity in snapshot.get('global_entities', []):
                entity.pop('points', None); entity.pop('colors', None)
            msg = String()
            msg.data = json.dumps(snapshot, allow_nan=False)
            self.state_pub.publish(msg)
            self.last_ros_publish = time.monotonic()

    def snapshot(self):
        now = time.monotonic()
        with self.state_lock:
            snapshot = deepcopy(self.cached)
            indexed = {r['robot_id']: r for r in snapshot.get('robots', [])}
            robots = []
            for settings in self.config['robots']:
                robot = settings['id']
                data = dict(indexed.get(robot, {}))
                data['mapping_status'] = data.get('status')
                if (data.get('session_id') and self.telemetry[robot].get('session_id')
                        and data['session_id'] != self.telemetry[robot]['session_id']):
                    data.update(aligned=False, T_world_local=None, keyframes=0, submaps=0,
                                mapping_status='new_session_pending')
                data.update(deepcopy(self.telemetry[robot]))
                advanced = data.get('health', {}).get('rgb_advanced_monotonic', data.get('last_frame_monotonic'))
                data['frame_age_s'] = None if advanced is None else round(now-advanced, 2)
                data['observation_age_s'] = None if 'last_processed_monotonic' not in data else round(now-data['last_processed_monotonic'], 2)
                data['timing'] = self.timing[robot].snapshot()
                data['timing_ms'] = data['timing']
                data['image_age_s'] = {view: round(now-stamp, 2) for (source, view), (_, stamp) in self.images.items() if source == robot}
                if data['frame_age_s'] is not None and data['frame_age_s'] > 5 and data['status'] not in ('adapter_exited', 'processing_error', 'input_rejected'):
                    data['status'] = 'input_stale'
                robots.append(data)
            if self.replay:
                for data in robots:
                    data.update(status='recorded', frame_age_s=None, observation_age_s=None,
                                image_age_s={key: None for key in data['image_age_s']})
            snapshot.update(robots=robots, mode='recorded_replay' if self.replay else 'synthetic_demo' if self.demo else 'live',
                run_dir=str(self.run_dir), central_domain=self.config['central_domain'],
                uptime_s=round(now-self.started, 1), timing=self.total_timing.snapshot(),
                active_robot=self.active_robot,
                processing_age_s=None if self.active_since is None else round(now-self.active_since, 1))
            snapshot['events'] = (snapshot.get('events', [])+list(self.events))[-80:]
        scheduler = self.scheduler.snapshot()
        fleet_health = self.fleet.health() if self.fleet else {}
        for data in snapshot['robots']:
            robot = data['robot_id']
            data['scheduler'] = scheduler[robot]
            data['perception'] = fleet_health.get(robot, {})
            if self.fleet and fleet_health.get(robot, {}).get('session_id') == data.get('session_id'):
                for view in ('sam', 'segments', 'entities', 'dino', 'dino_similarity', 'clip', 'depth'):
                    image = self.fleet.get_image(robot, view)
                    if image:
                        data['image_age_s'][view] = max(0., time.time()-image[1]/1e9)
        return snapshot

    def image(self, robot, view):
        if robot not in self.telemetry:
            return None
        if view != 'rgb' and self.fleet is not None:
            health = self.fleet.health().get(robot, {})
            with self.state_lock:
                current_session = self.telemetry[robot].get('session_id')
            result = self.fleet.get_image(robot, view) if health.get('session_id') == current_session else None
            if result:
                return result[0], 'image/jpeg'
        with self.state_lock:
            result = self.images.get((robot, view))
        return None if result is None else (result[0], 'image/jpeg')

    def command(self, command):
        if self.replay and command.get('action') == 'save':
            raise ValueError('Replay decisions are in-memory only; the source recording is read-only')
        with self.backend_lock:
            result = self.backend.command(command)
            snapshot = self.backend.snapshot()
            with self.state_lock:
                self.cached = snapshot
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        self.input_stop.set()
        self.dashboard.close()
        for adapter in self.adapters.values():
            adapter['process'].join(timeout=2)
            if adapter['process'].is_alive():
                adapter['process'].terminate()  # Only child adapters created by this run.
                adapter['process'].join(timeout=2)
            for name in ('output', 'health'):
                adapter[name].cancel_join_thread(); adapter[name].close()
        if self.worker:
            self.worker.join(timeout=10)
        busy = self.worker and self.worker.is_alive()
        if busy:
            print('Inference still finishing; exiting without destroying its ROS/model objects. Last completed evidence is on disk.', file=sys.stderr)
        else:
            if not self.replay:
                with self.backend_lock:
                    self.backend.command({'action': 'save'})
            if self.fleet:
                self.fleet.close()
            if self.publisher_node:
                self.publisher_node.destroy_node()
            if self.context:
                self.context.try_shutdown()


def check_environment(config, demo=False):
    from collab_backend import validate_config as validate_backend_config
    validate_backend_config(config.get('backend', {}))
    modules = ['numpy', 'cv2'] if demo else ['numpy', 'cv2', 'rclpy', 'tf2_ros', 'message_filters', 'torch', 'open_clip', 'efficient_sam']
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('Missing Python packages: '+', '.join(missing)+'. Activate the existing ros_humble_env environment.')
    if not demo:
        checkpoint = config['perception']['parameters'].get('esam_checkpoint', '')
        variant = config['perception']['parameters'].get('esam_variant', 'vitt')
        path = Path(checkpoint).expanduser() if checkpoint else ROOT.parent / 'weights' / f'efficient_sam_{variant}.pt'
        if not path.is_file():
            raise RuntimeError(f'eSAM checkpoint not found: {path}')
        from collab_perception import PerceptionFleet
        PerceptionFleet(config)  # Validate shared-model settings; loads no model or ROS node.
        if os.environ.get('ROS_LOCALHOST_ONLY') == '1':
            print('Warning: ROS_LOCALHOST_ONLY=1 prevents discovery of remote rovers; unset it for the live two-rover run.', file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='JSON config; defaults to the two-rover profile or saved run config on resume/replay')
    parser.add_argument('--run-dir', help='Child of experimental/.runtime/collaboration; a new directory by default')
    parser.add_argument('--resume', action='store_true', help='Resume evidence in --run-dir; new rover sessions are kept separate')
    parser.add_argument('--demo', action='store_true', help='Synthetic two-source demo, no ROS, models, or robot connections')
    parser.add_argument('--replay', metavar='RUN_DIR', help='Inspect a saved run without ROS/models or changing the recording')
    parser.add_argument('--check', action='store_true', help='Validate config/dependencies without connecting to robots')
    parser.add_argument('--port', type=int, help='Override loopback dashboard port (default 8766)')
    parser.add_argument('--duration', type=float, help='Stop after this many seconds (useful for tests)')
    args = parser.parse_args(argv)
    if args.resume and not args.run_dir:
        parser.error('--resume requires --run-dir')
    if args.replay and (args.demo or args.resume or args.run_dir):
        parser.error('--replay cannot be combined with --demo, --resume, or --run-dir')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')
    existing = owned_run_directory(args.replay or args.run_dir, resume=True) if args.replay or args.resume else None
    config_path = args.config or (existing/'config.json' if existing else ROOT/'config'/'collaborative_two_rovers.json')
    config = load_config(config_path)
    if existing and args.config:
        saved = load_config(existing/'config.json')
        for section in ('robots', 'perception', 'central_domain', 'input'):
            if config[section] != saved[section]:
                parser.error(f'Cannot change {section} while resuming/replaying; start a fresh run for changed sources/models')
    if args.port:
        from collab_config import validate_config
        config['dashboard']['port'] = args.port
        config = validate_config(config)
    check_environment(config, args.demo or bool(args.replay))
    if args.check:
        print(json.dumps({'status':'ready', 'central_domain':config['central_domain'],
                          'robots':[{k:r[k] for k in ('id','domain','local_frame','depth_registration')} for r in config['robots']],
                          'dashboard':config['dashboard'], 'mode':'replay' if args.replay else 'demo' if args.demo else 'live'}, indent=2))
        return 0
    run_dir = existing or owned_run_directory(args.run_dir)
    # Prefix keeps this guard separate from old v2 while preventing two fleet
    # owners of the same central-domain namespace, regardless of run directory.
    lock = ROOT/'.runtime'/(f'collaboration_review_{config["dashboard"]["port"]}.lock' if args.replay else f'collaboration_domain_{config["central_domain"]}.lock')
    with PipelineInstance(lock):
        run_dir.mkdir(parents=True, exist_ok=True)
        config_file = run_dir/'config.json'
        if not config_file.exists():
            config_file.write_text(json.dumps(config, indent=2)+'\n')
        os.environ.setdefault('ROS_LOG_DIR', str(run_dir/'ros_logs'))
        app = FleetApplication(config, run_dir, demo=args.demo, replay=bool(args.replay))
        shutdown = threading.Event()
        def stop(*_):
            shutdown.set()
        old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            app.start(live=not args.demo and not args.replay)
            mode = ('RECORDED REPLAY (source read-only)' if args.replay else 'SYNTHETIC DEMO (no robots)' if args.demo
                    else f'LIVE; automatic alignment {"SHADOW" if config["backend"].get("shadow_mode", True) else "ENABLED"}')
            print(f'CrossMaps collaborative: {app.dashboard.url}\nRun: {run_dir}\n'
                  f'Mode: {mode}\n'
                  'Ctrl-C stops only this server and its input adapters.', flush=True)
            started = last_demo = time.monotonic()
            demo_step = 0
            while not shutdown.is_set():
                app.poll_inputs()
                if args.demo and time.monotonic()-last_demo >= 1.:
                    from collab_demo import observations
                    for packet in observations(demo_step, config['robots']):
                        app.offer(packet)
                    demo_step += 1
                    last_demo = time.monotonic()
                if args.duration and time.monotonic()-started >= args.duration:
                    break
                shutdown.wait(.05)
        finally:
            app.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f'CrossMaps: {error}', file=sys.stderr)
        raise SystemExit(2)
