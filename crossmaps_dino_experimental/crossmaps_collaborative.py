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
import uuid

import cv2
import numpy as np

from collab_backend import Backend
from collab_config import CLIP_PRESETS, ROOT, configure_shared_clip, load_config, owned_run_directory
from collab_dashboard import CollaborativeDashboard
from collab_log_budget import ScopedLogBudget
from collab_runtime import FairLatestScheduler, CycleTiming
from collab_transport import require_compressed_transport, run_adapter
from pipeline_instance import PipelineInstance


class FleetApplication:
    """Owns all writes to backend state and exposes immutable UI snapshots."""
    def __init__(self, config, run_dir, *, demo=False, fleet=None, replay=False, storage=None, evaluation=None):
        self.config, self.run_dir, self.demo, self.replay = config, Path(run_dir), demo, replay
        self.storage = storage
        # Evaluation receives estimates only AFTER mapping. Truth is never
        # passed to adapters, perception, registration or the pose graph.
        self.evaluation = evaluation
        self.evaluation_snapshot = None
        self.next_evaluation = 0.
        self.mapping_lock = threading.RLock()
        self.map_generation = uuid.uuid4().hex
        self.map_epoch_started = 0.
        self.map_reset_error = None
        self.storage_plans = {}
        self.log_budget = ScopedLogBudget(self.run_dir) if storage and not replay else None
        if replay:
            if not (self.run_dir / 'evidence' / 'state.json').is_file():
                raise ValueError('Replay requires a completed collaborative evidence checkpoint')
            self.backend = Backend.replay(self.run_dir / 'evidence')
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            backend_options = dict(config['backend'])
            if (self.run_dir / 'evidence' / 'state.json').is_file():
                # A recording's config.json describes how it STARTED. The last
                # checkpoint has the actual operator decision at shutdown.
                backend_options.pop('shadow_mode', None)
            self.backend = Backend(backend_options, self.run_dir / 'evidence',
                                   evidence_budget=storage.check_write if storage else None)
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
        self.dashboard = CollaborativeDashboard(self.snapshot, self.command, self._image_response,
                                                identity_provider=self.identity_inspection,
                                                witness_provider=self.witness_image,
                                                **config['dashboard'])
        self.active_robot = None
        self.active_since = None
        self.publisher_node = self.state_pub = None
        self.started = time.monotonic()
        self.next_idle_verification = self.started + 8.
        self.last_ros_publish = 0.
        self._closed = False
        if replay:
            report = self.run_dir / 'evidence' / 'evaluation.json'
            if report.is_file() and not report.is_symlink() and report.stat().st_size < 1024**2:
                try:
                    value = json.loads(report.read_text())
                    if isinstance(value, dict) and value.get('schema') == 1:
                        self.evaluation_snapshot = dict(value, recorded=True)
                except (OSError, ValueError):
                    pass
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
        if self._simulation_reset_blocked(robot):
            return False
        with self.scheduler.lock:
            if observation['received_monotonic'] < self.map_epoch_started:
                return False  # Queued pre-rebuild captures cannot enter the new map.
            if not self.scheduler.offer(observation):
                return False
        received = observation['received_monotonic']
        with self.state_lock:
            state = self.telemetry[robot]
            if state.get('session_id') != observation['session_id']:
                for key in ('last_mapped_monotonic', 'map_finished_monotonic', 'shared_evidence_monotonic', 'last_processed_monotonic', 'last_processed_stamp_ns',
                            'pose_quality', 'counts', 'last_error'):
                    state.pop(key, None)
            # Live adapters supply camera previews asynchronously. Demo and direct
            # callers without an adapter use the offered frame as their preview.
            if robot not in self.adapters:
                ok, jpeg = cv2.imencode('.jpg', observation['rgb'], [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    self.images[(robot, 'rgb')] = (jpeg.tobytes(), received)
            state.update(last_frame_monotonic=received, session_id=observation['session_id'],
                         last_input_stamp_ns=observation['stamp_ns'],
                         status='processing' if self.active_robot == robot else 'receiving',
                         depth_registration=observation.get('depth_registration'),
                         calibration_id=observation.get('calibration_id'))
        return True

    def _simulation_reset_blocked(self, robot):
        return (self.config['input'].get('use_sim_time', False) and
                self.telemetry[robot].get('health', {}).get('simulation_clock_status') == 'reset_restart_required')

    def process_one(self):
        # Rebuild waits for this observation to finish, then clears both local
        # and shared state before the next observation can be processed.
        with self.mapping_lock:
            return self._process_one_locked()

    def _process_one_locked(self):
        if self.map_reset_error:
            return False  # A failed local reset must never mix old and new maps.
        observation = self.scheduler.take()
        if observation is None:
            return False
        robot = observation['robot_id']
        if self._simulation_reset_blocked(robot):
            return False
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
            if self._simulation_reset_blocked(robot):
                return False  # A world reset during inference cannot extend the old map.
            preview_only = observation.get('preview_only', False)
            retained = False
            if preview_only:
                with self.backend_lock:
                    if self.backend.mark_trajectory_gap(robot, observation['session_id']):
                        snapshot = self.backend.snapshot()
                        with self.state_lock:
                            self.cached = snapshot
            else:
                with self.backend_lock:
                    previous_count = len(self.backend.keyframes)
                    try:
                        snapshot = self.backend.ingest(enriched)
                    finally:
                        if self.storage:
                            self.storage.finish_write()
                    retained = len(self.backend.keyframes) != previous_count
                    if retained:
                        self.next_idle_verification = time.monotonic() + 8.
                    with self.state_lock:
                        self.cached = snapshot
            finished = time.monotonic()
            with self.state_lock:
                self.timing[robot].add(finished-started, inference_elapsed,
                                      finished-observation['received_monotonic'])
                self.total_timing.add(finished-started, inference_elapsed,
                                      finished-observation['received_monotonic'])
                state = self.telemetry[robot]
                if state.get('session_id') != observation['session_id']:
                    return True
                allowed = not preview_only and enriched.get('integration_allowed', True)
                if allowed:
                    state['last_mapped_monotonic'] = observation['received_monotonic']
                    # Input age and time since integration are different when
                    # waiting for measured TF or sharing the inference worker.
                    state['map_finished_monotonic'] = finished
                if retained:
                    state['shared_evidence_monotonic'] = finished
                status = 'waiting_for_tf' if preview_only else 'healthy' if allowed else 'pose_rejected'
                state.update(status=status,
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

    def _idle_verification(self):
        """Give one skipped pair a turn in a quiet live session. Never invent input."""
        now = time.monotonic()
        if (self.demo or self.replay or self.map_reset_error or self.stop_event.is_set() or now < self.next_idle_verification or
                any(q['pending'] for q in self.scheduler.snapshot().values())):
            return False
        self.next_idle_verification = now + 8.
        try:
            with self.backend_lock:
                with self.state_lock:
                    # A new adapter can arrive before its first valid pose.
                    # Don't confirm a previous session's previews in that gap.
                    for robot, sid in self.backend.robots.items():
                        if self.telemetry.get(robot, {}).get('session_id') != self.backend.sessions[sid]['session_id']:
                            return False
                if not self.backend.backfill_one():
                    return False
                snapshot = self.backend.snapshot()
                with self.state_lock:
                    self.cached = snapshot
                return True
        except Exception as error:
            self.event('verification_error', f'Idle verification: {type(error).__name__}: {error}')
            return False

    def _work(self):
        while not self.stop_event.is_set():
            if not self.process_one():
                self._idle_verification()
                self.stop_event.wait(.05)

    def _start_adapters(self, relay_passwords=None):
        for robot in self.config['robots']:
            output, health = self.mp_context.Queue(maxsize=1), self.mp_context.Queue(maxsize=2)
            process = self.mp_context.Process(target=run_adapter,
                args=(robot, self.config['input'], output, health, self.input_stop,
                      (relay_passwords or {}).get(robot['id'])),
                name=f"crossmaps-input-{robot['id']}", daemon=True)
            process.start()
            self.adapters[robot['id']] = dict(process=process, output=output, health=health)

    def start(self, live=True, relay_passwords=None):
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
                self._start_adapters(relay_passwords)
            if not self.replay:
                self.worker = threading.Thread(target=self._work, name='shared-perception', daemon=True)
                self.worker.start()
            return self
        except BaseException:
            self.close()
            raise

    def poll_inputs(self):
        if self.log_budget:
            self.log_budget.tick()
        self._refresh_evaluation()
        for robot, adapter in self.adapters.items():
            if not adapter['process'].is_alive():
                with self.state_lock:
                    self.telemetry[robot].update(status='adapter_exited', last_error=f"Input adapter exited ({adapter['process'].exitcode}); restart server after inspecting terminal")
                continue
            for _ in range(4):
                try:
                    health = adapter['health'].get_nowait()
                    with self.state_lock:
                        for view, prefix in (('rgb', 'preview'), ('depth', 'depth_preview')):
                            jpeg = health.pop(prefix+'_jpeg', None)
                            if jpeg:
                                self.images[(robot, view)] = (jpeg, health[prefix+'_monotonic'])
                        self.telemetry[robot]['health'] = health
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
            for rover in snapshot.get('robots', []):
                # Full trails belong to the HTTP spatial viewer, not the
                # periodic ROS diagnostics broadcast on the laptop network.
                if isinstance(rover.get('trajectory'), dict):
                    rover['trajectory'].pop('segments', None)
            for entity in snapshot.get('global_entities', []):
                entity.pop('points', None); entity.pop('colors', None)
            msg = String()
            msg.data = json.dumps(snapshot, allow_nan=False)
            self.state_pub.publish(msg)
            self.last_ros_publish = time.monotonic()

    def _refresh_evaluation(self):
        if self.evaluation is None or time.monotonic() < self.next_evaluation:
            return
        self.next_evaluation = time.monotonic() + 1.
        with self.backend_lock:
            try:
                report = self.evaluation.snapshot(self.backend)
                # One bounded report, not a growing per-frame log. Stored with
                # this map generation and removed by Rebuild current map.
                payload = json.dumps(report, allow_nan=False)
                if len(payload.encode()) > 1024**2:
                    raise ValueError('Evaluation report exceeded its size budget')
                path = self.run_dir / 'evidence' / 'evaluation.json'
                temporary = path.with_suffix('.json.tmp')
                temporary.write_text(payload)
                temporary.replace(path)
            except Exception as error:
                report = dict(enabled=True, schema=1, status='invalid',
                              reason=f'Evaluation unavailable: {error}')
            with self.state_lock:
                self.evaluation_snapshot = report

    def snapshot(self):
        now = time.monotonic()
        with self.state_lock:
            snapshot = deepcopy(self.cached)
            indexed = {r['robot_id']: r for r in snapshot.get('robots', [])}
            replaced = set()
            robots = []
            for settings in self.config['robots']:
                robot = settings['id']
                data = dict(indexed.get(robot, {}))
                data['mapping_status'] = data.get('status')
                if (data.get('session_id') and self.telemetry[robot].get('session_id')
                        and data['session_id'] != self.telemetry[robot]['session_id']):
                    replaced.add(robot)
                    data.update(aligned=False, alignment_state='unaligned', source_id=None,
                                T_world_local=None, T_world_camera=None, trajectory=None, keyframes=0, submaps=0,
                                mapping_status='new_session_pending')
                data.update(deepcopy(self.telemetry[robot]))
                advanced = data.get('health', {}).get('rgb_advanced_monotonic', data.get('last_frame_monotonic'))
                data['frame_age_s'] = None if advanced is None else round(now-advanced, 2)
                data['observation_age_s'] = None if 'last_processed_monotonic' not in data else round(now-data['last_processed_monotonic'], 2)
                mapped = data.get('last_mapped_monotonic')
                data['map_observation_age_s'] = None if mapped is None else round(now-mapped, 2)
                finished = data.get('map_finished_monotonic')
                data['map_update_age_s'] = None if finished is None else round(now-finished, 2)
                data['local_map_update_age_s'] = data['map_update_age_s']
                retained = data.get('shared_evidence_monotonic')
                data['shared_evidence_age_s'] = None if retained is None else round(now-retained, 2)
                data['evidence_status'] = data.get('evidence', {}).get('outcome')
                data['timing'] = self.timing[robot].snapshot()
                data['timing_ms'] = data['timing']
                data['image_age_s'] = {view: round(now-stamp, 2) for (source, view), (_, stamp) in self.images.items() if source == robot}
                sim_clock = data.get('health', {}).get('simulation_clock_status')
                if (self.config['input'].get('use_sim_time', False) and sim_clock in
                        ('waiting_for_clock', 'paused_or_stalled', 'reset_restart_required') and
                        data['status'] not in ('adapter_exited', 'processing_error')):
                    data['status'] = 'simulation_'+sim_clock
                elif data['frame_age_s'] is not None and data['frame_age_s'] > 5 and data['status'] not in ('adapter_exited', 'processing_error', 'input_rejected'):
                    data['status'] = 'input_stale'
                robots.append(data)
            # A new adapter session can arrive before its first accepted RGB-D
            # pose. Don't render a cached preview against either old endpoint.
            # Keep the recorded, confirmed map history and backend untouched.
            if replaced:
                evidence = snapshot['alignment_evidence']
                for field in ('provisional', 'provisional_diagnostics'):
                    evidence[field] = [r for r in evidence.get(field, [])
                                       if not replaced.intersection((r['source_robot'], r['target_robot']))]
                visible = {r['source_robot'] for r in evidence['provisional']}
                for item in [*robots, *snapshot['submaps']]:
                    if item.get('alignment_state') == 'provisional' and item['robot_id'] not in visible:
                        item['alignment_state'] = 'unaligned'
                        if 'T_preview_world_submap' in item:
                            item['T_preview_world_submap'] = None
            if self.replay:
                for data in robots:
                    data.update(status='recorded', frame_age_s=None, observation_age_s=None,
                                image_age_s={key: None for key in data['image_age_s']})
            snapshot.update(robots=robots, mode='recorded_replay' if self.replay else 'synthetic_demo' if self.demo else 'live',
                run_dir=str(self.run_dir), map_generation=self.map_generation, central_domain=self.config['central_domain'],
                uptime_s=round(now-self.started, 1), timing=self.total_timing.snapshot(),
                active_robot=self.active_robot,
                processing_age_s=None if self.active_since is None else round(now-self.active_since, 1))
            snapshot['events'] = (snapshot.get('events', [])+list(self.events))[-80:]
            snapshot['simulation_evaluation'] = deepcopy(self.evaluation_snapshot)
        scheduler = self.scheduler.snapshot()
        fleet_health = self.fleet.health() if self.fleet and not self.map_reset_error else {}
        for data in snapshot['robots']:
            robot = data['robot_id']
            data['scheduler'] = scheduler[robot]
            health = fleet_health.get(robot, {})
            session = data.get('session_id')
            data['perception'] = health if health.get('session_id') == session else {}
            if self.map_reset_error:
                data.update(status='processing_error', mapping_status='reset_failed',
                            last_error=self.map_reset_error, perception={})
                data['image_age_s'] = {key: age for key, age in data['image_age_s'].items()
                                       if key in ('rgb', 'depth')}
            elif self.fleet and session:
                for view, received in self.fleet.image_metadata(robot, session).items():
                    data['image_age_s'].setdefault(view, round(max(0., now-received), 2))
        snapshot['storage'] = self.storage.inventory() if self.storage else None
        return snapshot

    def image(self, robot, view):
        response = self._image_response(robot, view)
        return response[:2] if response is not None else None

    def _image_response(self, robot, view):
        if robot not in self.telemetry:
            return None
        if self.map_reset_error and view not in ('rgb', 'depth'):
            return None
        with self.state_lock:
            result = self.images.get((robot, view))
            session = self.telemetry[robot].get('session_id')
            if result is None and self.fleet and session:
                result = self.fleet.image_with_receipt(robot, view, session_id=session)
        if result is None:
            return None
        return (result[0], 'image/jpeg', None if self.replay else max(0., time.monotonic() - result[1]))

    def identity_inspection(self, link):
        with self.backend_lock:
            return self.backend.identity_inspection(link)

    def witness_image(self, link, member, frame):
        with self.backend_lock:
            return self.backend.identity_evidence_image(link, member, frame)

    def command(self, command):
        if command.get('action') in ('prepare_storage', 'confirm_storage'):
            return self._storage_command(command)
        if self.replay and command.get('action') == 'save':
            raise ValueError('Replay decisions are in-memory only; the source recording is read-only')
        with self.backend_lock:
            result = self.backend.command(command)
            snapshot = self.backend.snapshot()
            with self.state_lock:
                self.cached = snapshot
        return result

    def _storage_command(self, command):
        if self.replay or self.storage is None:
            raise ValueError('Storage changes require a live, managed recording; replay is read-only')
        if command['action'] == 'prepare_storage':
            operation = command.get('operation')
            if operation == 'rebuild_current':
                plan = self.storage.prepare_reset()
                description = ('Permanently delete this run’s retained map evidence and clear its local maps, '
                               'alignments, identities and paths. Rebuild from new camera observations. '
                               'Robot SLAM, Gazebo and navigation are not reset.')
            elif operation == 'delete_runs':
                plan = self.storage.prepare_cleanup(command.get('run_ids', []))
                description = 'Permanently delete only the selected inactive recordings. This cannot be undone.'
            else:
                raise ValueError('Unknown storage operation')
            self.storage_plans = {plan['token']: (operation, self.map_generation)}
            runs = plan.get('run_ids', plan.get('runs', []))
            run_ids = [r['id'] if isinstance(r, dict) else r for r in runs]
            if operation == 'rebuild_current':
                run_ids = [self.run_dir.name]
            return dict(ok=True, plan=dict(token=plan['token'], operation=operation,
                        run_ids=run_ids, bytes=plan['bytes'], description=description))
        token = command.get('token')
        if not isinstance(token, str) or token not in self.storage_plans:
            raise ValueError('Prepare a storage operation before confirming it')
        operation, generation = self.storage_plans.pop(token)
        with self.mapping_lock:
            if generation != self.map_generation:
                raise ValueError('The active map changed; review the operation again')
            if operation == 'delete_runs':
                result = self.storage.cleanup(token)
                complete = result.get('ok', True)
                message = (f"Removed {len(result.get('removed', []))} selected recordings; "
                           f"freed {result.get('freed_bytes', 0)} bytes.")
                if not complete:
                    message += ' ' + result.get('error', 'Cleanup was only partially completed.')
                self.event('storage', message)
                return dict(ok=complete, error=None if complete else message, message=message,
                            operation=operation, result=result, storage=self.storage.inventory(force=True))
            # Delete only after an explicit, previewed confirmation. Captures
            # acquired before this boundary are discarded, even if still queued
            # in an input adapter. Shared neural weights remain loaded.
            with self.backend_lock:
                options = dict(self.backend.config)
                # Validate and allocate before deleting anything. Once the disk
                # reset commits, no later failure may restore the old graph.
                replacement = Backend(options, evidence_budget=self.storage.check_write)
                result = self.storage.reset_current(token)
                self.backend = replacement
                self.backend.store_dir = self.run_dir / 'evidence'
                if self.evaluation:
                    self.evaluation.reset()
                self.evaluation_snapshot = None
                self.next_evaluation = 0.
                self.map_generation = uuid.uuid4().hex
                self.map_reset_error = None
                with self.scheduler.lock:
                    self.map_epoch_started = time.monotonic()
                    self.scheduler.pending.clear()
                with self.state_lock:
                    self.cached = self.backend.snapshot()
                    for state in self.telemetry.values():
                        for key in ('last_mapped_monotonic', 'map_finished_monotonic', 'shared_evidence_monotonic',
                                    'last_processed_monotonic', 'last_processed_stamp_ns', 'pose_quality', 'counts', 'last_error'):
                            state.pop(key, None)
                        state['status'] = 'waiting_for_input'
                    self.images = {key: value for key, value in self.images.items() if key[1] in ('rgb', 'depth')}
                warnings = [result['warning']] if result.get('warning') else []
                try:
                    if self.fleet:
                        self.fleet.reset_maps()
                except Exception as error:
                    self.map_reset_error = f'Local map reset failed; rebuild again or restart: {error}'
                    warnings.append(self.map_reset_error)
                    with self.state_lock:
                        for state in self.telemetry.values():
                            state.update(status='processing_error', last_error=self.map_reset_error)
                self.next_idle_verification = time.monotonic() + 8.
                try:
                    self.backend.command({'action': 'save'})
                except Exception as error:
                    warnings.append(f'Map was cleared, but its empty checkpoint could not be saved: {error}')
                if warnings:
                    result['warning'] = ' '.join(warnings)
            self.event('map_rebuilt', 'Current map evidence cleared by operator; rebuilding from fresh observations')
            return dict(ok=True, operation=operation, result=result, storage=self.storage.inventory(force=True))

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
        if self.storage and not busy:
            self.storage.close()
        # A still-running inference worker may finish writing after close().
        # Keep the recording's ownership lock until process exit in that case.


def check_environment(config, demo=False):
    from collab_backend import validate_config as validate_backend_config
    validate_backend_config(config.get('backend', {}))
    if not demo:
        for robot in config['robots']:
            require_compressed_transport(robot)
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
        if os.environ.get('ROS_LOCALHOST_ONLY') == '1' and not config['input'].get('use_sim_time', False):
            print('Warning: ROS_LOCALHOST_ONLY=1 prevents discovery of remote rovers; unset it for the live two-rover run.', file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='JSON config; defaults to the two-rover profile or saved run config on resume/replay')
    parser.add_argument('--run-dir', help='Child of experimental/.runtime/collaboration; a new directory by default')
    parser.add_argument('--resume', action='store_true', help='Resume evidence in --run-dir; new rover sessions are kept separate')
    parser.add_argument('--demo', action='store_true', help='Synthetic two-source demo, no ROS, models, or robot connections')
    parser.add_argument('--shadow-mode', action='store_true',
                        help='Start a new run in review-only mode for cross-rover decisions (live default: automatic alignment)')
    parser.add_argument('--replay', metavar='RUN_DIR', help='Inspect a saved run without ROS/models or changing the recording')
    parser.add_argument('--check', action='store_true', help='Validate config/dependencies without connecting to robots')
    parser.add_argument('--clip-preset', choices=CLIP_PRESETS,
                        help='Select the shared image/text encoder for a new run (or --check). '
                             'Omit to use the selected config; vit-l-14 restores the OpenAI baseline.')
    parser.add_argument('--port', type=int, help='Override loopback dashboard port (default 8766)')
    parser.add_argument('--duration', type=float, help='Stop after this many seconds (useful for tests)')
    parser.add_argument('--simulation-truth', type=Path, help='Evaluation-only Gazebo truth sidecar (never a mapping input)')
    parser.add_argument('--simulation-scene', type=Path, help='Generated simulation scene containing optical extrinsics')
    parser.add_argument('--prompt-relay-password', action='store_true',
                        help='Force a private, session-only SSH password prompt instead of automatic key/Keychain lookup')
    parser.add_argument('--setup-relay-auth', action='store_true',
                        help='One-time macOS setup: verify access and save host-scoped camera recovery credentials to Keychain, then exit')
    args = parser.parse_args(argv)
    if args.resume and not args.run_dir:
        parser.error('--resume requires --run-dir')
    if args.replay and (args.demo or args.resume or args.run_dir):
        parser.error('--replay cannot be combined with --demo, --resume, or --run-dir')
    if args.shadow_mode and (args.resume or args.replay or args.setup_relay_auth):
        parser.error('--shadow-mode is a startup option for new runs; existing runs retain their checkpoint mode')
    if args.clip_preset and (args.resume or args.replay or args.setup_relay_auth):
        parser.error('--clip-preset is only for new runs or their configuration check; '
                     'saved runs retain their recorded encoder')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')
    existing = owned_run_directory(args.replay or args.run_dir, resume=True) if args.replay or args.resume else None
    config_path = args.config or (existing/'config.json' if existing else ROOT/'config'/'collaborative_two_rovers.json')
    config = load_config(config_path)
    if bool(args.simulation_truth) != bool(args.simulation_scene):
        parser.error('--simulation-truth and --simulation-scene must be supplied together')
    evaluation = None
    if args.simulation_truth:
        if (args.demo or args.resume or args.replay or args.setup_relay_auth
                or not config['input'].get('use_sim_time') or config['central_domain'] != 82
                or {(r['id'], r['domain']) for r in config['robots']} != {('rover_a', 80), ('rover_b', 81)}):
            parser.error('Ground truth evaluation is only available for fresh simulation runs on domains 80/81/82')
        from collab_evaluation import SimulationEvaluator
        evaluation = SimulationEvaluator(args.simulation_truth, args.simulation_scene)
    try:
        clip = configure_shared_clip(config, args.clip_preset)
    except ValueError as error:
        parser.error(str(error))
    if args.shadow_mode or (args.demo and not existing and not args.config):
        # Preserve the review-first synthetic walkthrough without changing a
        # user's explicit demo profile or the real two-rover default.
        config['backend']['shadow_mode'] = True
    if (args.setup_relay_auth or args.prompt_relay_password) and (args.demo or args.replay or args.check):
        parser.error('Relay credentials apply only to live mode')
    if args.setup_relay_auth and (args.prompt_relay_password or args.run_dir or args.resume or args.duration or args.port):
        parser.error('--setup-relay-auth only accepts an optional --config and never starts a coordinator')
    def relay_access():
        from collab_relay_auth import MacKeychain, PASSWORD_ENV, prepare
        # This application never exports a credential. Pass only the relevant
        # per-source password through the private multiprocessing spawn pipe.
        return prepare(config['robots'], setup=args.setup_relay_auth, force_prompt=args.prompt_relay_password,
                       store=MacKeychain() if sys.platform=='darwin' else None,
                       interactive=sys.stdin.isatty(), session_password=os.environ.pop(PASSWORD_ENV,None))
    if args.setup_relay_auth:
        relay_access()
        return 0
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
                          'clip':{'model_name':clip['clip_model_name'], 'pretrained':clip['clip_pretrained'],
                                  'weights_checked':False},
                          'robots':[{k:r[k] for k in ('id','domain','local_frame','base_frame','depth_registration')} for r in config['robots']],
                          'dashboard':config['dashboard'], 'mode':'replay' if args.replay else 'demo' if args.demo else 'live',
                          'startup_alignment':'from checkpoint' if existing else
                              ('shadow' if config['backend']['shadow_mode'] else 'automatic')}, indent=2))
        return 0
    run_dir = existing or owned_run_directory(args.run_dir)
    # Prefix keeps this guard separate from old v2 while preventing two fleet
    # owners of the same central-domain namespace, regardless of run directory.
    lock = ROOT/'.runtime'/(f'collaboration_review_{config["dashboard"]["port"]}.lock' if args.replay else f'collaboration_domain_{config["central_domain"]}.lock')
    with PipelineInstance(lock):
        relay_passwords = relay_access() if not (args.replay or args.demo) else {}
        run_dir.mkdir(parents=True, exist_ok=True)
        config_file = run_dir/'config.json'
        if not config_file.exists():
            config_file.write_text(json.dumps(config, indent=2)+'\n')
        os.environ.setdefault('ROS_LOG_DIR', str(run_dir/'ros_logs'))
        from collab_storage import StorageManager
        storage = StorageManager(ROOT / '.runtime' / 'collaboration', run_dir,
                                 readonly=bool(args.replay), owned_domain=config['central_domain'],
                                 domain_lock_held=not bool(args.replay), **config['storage'])
        try:
            app = FleetApplication(config, run_dir, demo=args.demo, replay=bool(args.replay), storage=storage,
                                   evaluation=evaluation)
        except Exception:
            storage.close()
            raise
        shutdown = threading.Event()
        def stop(*_):
            shutdown.set()
        old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            app.start(live=not args.demo and not args.replay, relay_passwords=relay_passwords)
            relay_passwords.clear()  # individual adapters now own only their session credential
            mode = ('RECORDED REPLAY (source read-only)' if args.replay else 'SYNTHETIC DEMO (no robots)' if args.demo
                    else f'LIVE; automatic alignment {"SHADOW" if app.backend.config["shadow_mode"] else "ENABLED"}')
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
