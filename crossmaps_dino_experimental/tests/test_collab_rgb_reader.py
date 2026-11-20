import unittest
import multiprocessing
import os
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from collab_transport import (CameraPath, CompressedWatchdog, PendingPair, RGBReaderWatch,
                             RebindableRGB, compressed_writer_ids, offer_relay_pair, stamp_ns)


class RGBReaderRecoveryTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv('CROSSMAPS_ROS_SMOKE') == '1', 'localhost ROS integration')
    def test_adapter_relay_ingress_emits_calibrated_timestamped_packet_but_never_replays_cached_pair(self):
        # Real adapter, ROS TF buffer, JPEG + compressedDepth, and process IPC.
        # The transport alone is fake: no robot, SSH connection, or raw IMAGE topics.
        self.assertEqual(os.getenv('ROS_LOCALHOST_ONLY'), '1')
        import cv2
        import numpy as np
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage
        from tf2_ros import TransformBroadcaster
        from collab_config import validate_config
        from collab_transport import run_adapter
        from test_collab_transport_ros import AdapterRosSmokeTests as util
        from unittest.mock import patch
        cfg=validate_config(dict(central_domain=88, robots=[dict(id='relay_a',domain=89,
            camera_relay=dict(mode='always',ssh_host='example.invalid')), dict(id='relay_b',domain=90)]))
        robot=cfg['robots'][0]; cfg['input']['max_hz']=10.
        packets, health = multiprocessing.Queue(2), multiprocessing.Queue(2)
        incoming, stop = queue.SimpleQueue(), threading.Event()
        relay = Mock()
        relay.poll.side_effect=lambda: None if incoming.empty() else incoming.get_nowait()
        relay.status.return_value={}
        ctx=Context()
        rclpy.init(args=[],context=ctx,domain_id=89)
        node=Node('relay_smoke_source',context=ctx)
        tf=TransformBroadcaster(node)
        adapter=threading.Thread(target=run_adapter,args=(robot,cfg['input'],packets,health,stop),daemon=True)
        try:
            with patch('collab_camera_relay.CameraRelay',return_value=relay):
                adapter.start()
                deadline=time.monotonic()+8.
                while not (relay.start.called and tf.pub_tf.get_subscription_count()) and time.monotonic()<deadline:
                    time.sleep(.05)
                self.assertTrue(relay.start.called and tf.pub_tf.get_subscription_count())
                self.assertEqual(node.count_subscribers(robot['rgb_topic']),0)
                self.assertEqual(node.count_subscribers(robot['depth_topic']),0)
                ns=node.get_clock().now().nanoseconds
                stamp=util._stamp(ns-100_000_000)
                rgb,depth=CompressedImage(),CompressedImage()
                for msg in (rgb,depth):
                    msg.header.frame_id='camera'
                    msg.header.stamp=stamp
                rgb.format='jpeg'; rgb.data=cv2.imencode('.jpg',np.full((32,48,3),110,np.uint8))[1].tobytes()
                depth.format='16UC1; compressedDepth png'
                depth.data=bytes(12)+cv2.imencode('.png',np.full((32,48),1400,np.uint16))[1].tobytes()
                pair=SimpleNamespace(rgb=rgb,depth=depth,rgb_info=util._info(stamp),depth_info=util._info(stamp),
                                     generation=1,received_wall_ns=ns,received_monotonic=time.monotonic())
                tf.sendTransform([util._tf(stamp,'odom','camera',[1.,2.,.3]),
                                  util._tf(util._stamp(ns),'odom','camera',[9.,9.,9.])])
                time.sleep(.07)
                incoming.put(pair)
                packet=packets.get(timeout=5.)
                self.assertEqual((packet['rgb_stamp_ns'],packet['stamp_ns']),(stamp_ns(stamp),stamp_ns(stamp)))
                self.assertFalse(packet['preview_only'])
                np.testing.assert_allclose(packet['T_local_camera'][:3,3],[1.,2.,.3])
                np.testing.assert_allclose(packet['depth'],1.4,atol=1e-5)
                incoming.put(pair)
                incoming.put(SimpleNamespace(**dict(vars(pair),generation=2,
                                                     received_monotonic=time.monotonic()-30.)))
                with self.assertRaises(queue.Empty):
                    packets.get(timeout=.45)
        finally:
            stop.set()
            if adapter.is_alive():
                adapter.join(timeout=4.)
            node.destroy_node(); ctx.try_shutdown()
            for channel in (packets,health):
                channel.cancel_join_thread(); channel.close()
        self.assertFalse(adapter.is_alive())

    def test_both_stalled_without_writers_wait_then_rebind_on_rediscovery(self):
        t = [0.]
        watch = CompressedWatchdog(2., lambda:t[0]); bind = Mock()
        writers = dict(rgb={'a'}, depth={'b'})
        watch.poll(writers, bind)
        t[0] = 15.
        watch.poll(dict(rgb=set(), depth=set()), bind)
        bind.assert_not_called()
        self.assertEqual(watch.health()['status'], 'waiting_for_camera_publishers')
        watch.poll(dict(rgb={'c'}, depth={'d'}), bind)
        bind.assert_called_once_with(('rgb','depth'))
        self.assertEqual(watch.health()['status'], 'rgbd_stale')
        watch.streams['rgb'].observe(10, .5)
        self.assertEqual(watch.health()['status'], 'depth_stale')
        self.assertEqual(watch.streams['depth'].observe(10, 200.), 'drop')
        t[0] += 2.
        watch.poll(dict(rgb={'e'}, depth={'f'}), bind)  # GID churn respects cooldown.
        self.assertEqual(bind.call_count, 1)
        watch.streams['depth'].observe(10, .2)
        self.assertNotIn('status', watch.health())

    def test_depth_only_rebind_keeps_rgb_reader_and_drops_late_depth_callback(self):
        t = [0.]
        watch = CompressedWatchdog(2., lambda:t[0]); callbacks = {'rgb':[], 'depth':[]}
        retired, signals = [], []
        def subscribe(kind, callback):
            callbacks[kind].append(callback)
            return callback
        def sync():
            return (lambda msg:signals.append(('rgb',msg)), lambda msg:signals.append(('depth',msg)))
        readers = RebindableRGB(lambda cb:subscribe('rgb',cb), retired.append, sync, lambda m,k:True,
                              lambda:None, lambda cb:subscribe('depth',cb))
        readers.rebind()
        t[0] = 14.
        watch.streams['rgb'].observe(10, .1)
        watch.poll(dict(rgb={'a'}, depth={'b'}), readers.rebind)
        self.assertEqual(retired, [callbacks['depth'][0]])
        callbacks['depth'][0]('retired'); callbacks['rgb'][0]('r'); callbacks['depth'][1]('d')
        self.assertEqual(signals, [('rgb','r'), ('depth','d')])

    def test_changed_writer_detected_without_payload_and_old_stamp_cannot_recover(self):
        t = [0.]
        watch = CompressedWatchdog(2., lambda:t[0]); bind = Mock()
        w = watch.streams['rgb']
        w.observe(8_000_000_000,.1)
        watch.poll(dict(rgb={'x'},depth={'y'}), bind)
        t[0] = 2.
        watch.poll(dict(rgb={'new'},depth={'y'}), bind)
        bind.assert_called_once_with(('rgb',))
        self.assertEqual(w.observe(8_000_000_000,.1), 'drop')
        self.assertEqual(w.observe(1_000_000_000,.1), 'drop')  # one skewed/reordered header is not a new epoch
        t[0] += .15
        self.assertEqual(w.observe(1_200_000_000,.1), 'drop')
        t[0] += .15
        self.assertEqual(w.observe(1_300_000_000,.1), 'advance')
        self.assertFalse(w.awaiting)

    def test_graph_endpoint_reporting_is_compact_and_handles_unsupported_rmw(self):
        graph = Mock()
        graph.get_publishers_info_by_topic.return_value=[SimpleNamespace(endpoint_gid=bytearray([2,3]))]
        self.assertEqual(compressed_writer_ids(graph,'/rgb/compressed'),{b'\x02\x03'})
        graph.get_publishers_info_by_topic.side_effect=NotImplementedError
        self.assertIsNone(compressed_writer_ids(graph,'/rgb/compressed'))

    def test_auto_relay_uses_one_transport_and_original_stamps_info_and_receipt(self):
        t = [0.]; clock = lambda:t[0]
        readers, relay = Mock(), Mock()
        relay.status.return_value = {'last_error': None}
        pending = PendingPair(2.,clock)
        watch = CompressedWatchdog(2.,clock)
        endpoints = dict(rgb={'rgb-gid'}, depth={'depth-gid'})
        def message(s):
            return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=s,nanosec=0)))
        def seen(msg, kind):
            return watch.streams[kind].observe(stamp_ns(msg.header.stamp), .1) == 'advance'
        def deliver(pair):
            return offer_relay_pair(pair, seen, pending, 2., .1, clock())
        camera = CameraPath('auto', readers, lambda:relay, deliver, pending.clear,clock)
        camera.start()
        readers.rebind.assert_called_once_with()
        camera.watch(watch,endpoints)
        t[0]=13.
        watch.streams['depth'].observe(10_000_000_000,.1)
        camera.watch(watch,endpoints)
        self.assertEqual(readers.rebind.call_args.args,(('rgb',),))
        relay.start.assert_not_called()
        t[0]=16.
        watch.streams['depth'].observe(22_000_000_000,.1)
        camera.watch(watch,endpoints)
        readers.close.assert_not_called()  # Healthy depth continues during SSH login.
        relay.start.assert_called_once()
        self.assertEqual(camera.health()['camera_transport'],'dds_compressed')
        self.assertTrue(camera.health()['camera_relay_probing'])
        t[0]=25.
        rgb,depth = message(25),message(25)
        info,di = object(),object()
        pair = SimpleNamespace(received_monotonic=24.8,received_wall_ns=25_200_000_000,
                               generation=1,rgb=rgb,depth=depth,rgb_info=info,depth_info=di)
        relay.poll.return_value=pair
        camera.poll()
        readers.close.assert_called_once()
        self.assertEqual(camera.health()['camera_transport'],'ssh_compressed_pair')
        r,d,received,meta = pending.take()
        self.assertIs(r,rgb); self.assertIs(d,depth); self.assertEqual(received,24.8)
        self.assertEqual(meta['received_wall_ns'],25_200_000_000)
        self.assertEqual(meta['calibration'],dict(rgb=info, depth=di))
        self.assertIsNone(meta.get('relay_pose'))
        pending.done()
        camera.poll()
        self.assertIsNone(pending.take())  # duplicate tunneled images never become a new local observation
        t[0]=28.; pair.generation=2
        camera.poll()
        self.assertIsNone(pending.take())  # a reconnect or poll cannot freshen an old packet
        relay.poll.return_value=None
        t[0]=38.
        camera.poll()
        relay.close.assert_not_called()
        self.assertEqual(camera.health()['camera_transport'],'dds_compressed')
        self.assertTrue(camera.health()['camera_relay_probing'])
        camera.watch(watch,endpoints)
        # Even long after the old 45 s teardown, the same connection can
        # deliver a fresh capture immediately (no 60 s forced DDS-only gap).
        t[0]=78.
        camera.poll()
        relay.close.assert_not_called()
        relay.poll.return_value = SimpleNamespace(received_monotonic=77.8,received_wall_ns=78_200_000_000,
            generation=2,rgb=message(78),depth=message(78),rgb_info=info,depth_info=di)
        camera.poll()
        self.assertEqual(camera.health()['camera_transport'],'ssh_compressed_pair')
        self.assertEqual(stamp_ns(pending.take()[0].header.stamp),78_000_000_000)
        self.assertEqual(relay.start.call_count,1)

    def test_auth_failure_never_retires_or_recreates_healthy_dds(self):
        now = [0.]; clock = lambda: now[0]
        readers, relay, deliver = Mock(), Mock(), Mock()
        from collab_camera_relay import _failure
        relay.status.return_value = {'last_error': _failure(b'Permission denied (publickey,password)')}
        relay.poll.return_value = None
        path = CameraPath('auto', readers, lambda:relay, deliver, Mock(), clock)
        watch = CompressedWatchdog(2.,clock)
        writers = dict(rgb={'r'},depth={'d'})
        path.start()
        now[0] = 25.
        watch.streams['depth'].observe(10,.1)
        path.watch(watch, writers)
        self.assertTrue(path.health()['camera_relay_probing'])
        path.poll()
        self.assertFalse(path.health()['camera_relay_probing'])
        self.assertIn('login denied',path.health()['camera_relay_last_error'])
        readers.close.assert_not_called()
        self.assertEqual(readers.rebind.call_count,2)  # Initial bind + stalled RGB only.
        now[0] = 100.
        watch.streams['depth'].observe(20,.1)
        path.watch(watch, writers)
        self.assertEqual(relay.start.call_count,1)
        deliver.assert_not_called()

    def test_cancel_pending_fallback_if_dds_recovers(self):
        now = [0.]; clock=lambda:now[0]
        readers, relay = Mock(), Mock()
        path = CameraPath('auto',readers,lambda:relay,Mock(),Mock(),clock)
        watch = CompressedWatchdog(2.,clock)
        path.start(); now[0]=25.
        path.watch(watch,dict(rgb={'r'},depth={'d'}))
        for w in watch.streams.values():
            w.observe(10,.1)
        path.watch(watch,dict(rgb={'r'},depth={'d'}))
        relay.close.assert_called_once_with(wait=False)
        self.assertIsNone(path.health()['camera_relay_last_error'])
        readers.close.assert_not_called()

    def test_missing_dds_discovery_does_not_mask_an_available_onboard_camera(self):
        now = [0.]; clock = lambda: now[0]
        readers, relay = Mock(), Mock()
        path = CameraPath('auto',readers,lambda:relay,Mock(),Mock(),clock)
        watch = CompressedWatchdog(2.,clock)
        absent = dict(rgb=set(),depth=set())
        path.start(); now[0] = 40.; path.watch(watch, absent)
        relay.start.assert_not_called()
        now[0] = 46.; path.watch(watch, absent); path.watch(watch, absent)
        relay.start.assert_called_once_with()
        relay.close.assert_not_called()
        readers.close.assert_not_called()

    def test_only_repeated_tf_expirations_probe_and_only_stamped_pose_selects_ssh(self):
        now = [0.]; clock = lambda:now[0]
        readers, relay, deliver = Mock(), Mock(), Mock(return_value=True)
        path = CameraPath('auto',readers,lambda:relay,deliver,Mock(),clock)
        watch = CompressedWatchdog(2.,clock)
        writers = dict(rgb={'r'},depth={'d'})
        path.start()
        def sample(t, expired, last_pose=None):
            now[0] = t
            for w in watch.streams.values(): w.observe(int(t*1e9), .2)
            path.watch(watch,writers,expired,last_pose)
        sample(1,0,1); sample(10,1,1); sample(16,1,1)
        relay.start.assert_not_called()
        sample(17,2,1)
        self.assertEqual(path.health()['camera_relay_probe_reason'],'tf')
        relay.start.assert_called_once()
        relay.poll.return_value = SimpleNamespace(generation=1,pose=None)
        relay.status.return_value = {}
        path.poll(); sample(19,2,1); path.poll()
        relay.close.assert_not_called(); readers.close.assert_not_called()
        deliver.assert_not_called()
        # The original DDS path recovers without any disruption or cooldown of TF.
        sample(20,2,20)
        relay.close.assert_called_once_with(wait=False)
        sample(35,3,20); self.assertIsNone(path.relay)
        sample(37,4,20)
        self.assertIs(path.relay,relay)
        relay.poll.return_value = SimpleNamespace(generation=1,pose={'matrix': 'validated elsewhere'})
        path.poll()
        readers.close.assert_called_once(); deliver.assert_called_once()
        self.assertEqual(path.health()['camera_transport'],'ssh_compressed_pair')

    def test_stalled_rgb_requires_advancing_depth_backoff_and_actual_new_source_stamp(self):
        t = [0.]
        watch = RGBReaderWatch(2., clock=lambda: t[0])
        open_reader = Mock()
        self.assertEqual(watch.observe(10_000_000_000, .3), 'advance')
        t[0] = 11.
        watch.poll(11., .3, open_reader)
        t[0] = 13.
        self.assertEqual(watch.observe(10_000_000_000, .4), 'unchanged')  # arriving cached RGB
        for depth_progress, header_age in [(None, .3), (7., .2), (13., 4.)]:
            watch.poll(depth_progress, header_age, open_reader)
        open_reader.assert_not_called()
        watch.poll(13., .3, open_reader)
        open_reader.assert_called_once()
        self.assertEqual(watch.health()['status'], 'rgb_stale')
        self.assertEqual(watch.observe(10_000_000_000, .4), 'drop')  # duplicate
        self.assertEqual(watch.observe(10_000_000_001, 9.), 'drop')  # stale on arrival
        self.assertEqual(watch.observe(9_999_999_999, .3), 'drop')  # old queued frame
        t[0] = 41.
        watch.poll(41., .1, open_reader)
        open_reader.assert_called_once()
        t[0] = 43.
        watch.poll(43., .2, open_reader)
        self.assertEqual(open_reader.call_count, 2)
        self.assertEqual(watch.observe(10_100_000_000, .1), 'advance')
        self.assertEqual(watch.recoveries, 1)
        self.assertFalse(watch.awaiting)
        self.assertNotIn('status', watch.health())  # do not override an unrelated TF error
        t[0] += 2.
        watch.poll(45., .1, open_reader)
        self.assertEqual(open_reader.call_count, 2)

    def test_failed_rebind_keeps_watchdog_and_subsequent_recovery(self):
        t = [0.]
        watch = RGBReaderWatch(2., clock=lambda: t[0])
        t[0] = 40.
        failed = Mock(side_effect=RuntimeError('DDS refused'))
        watch.poll(39., .2, failed)
        self.assertIn('DDS refused', watch.health()['last_error'])
        t[0] += 1
        watch.poll(41., .2, failed)
        failed.assert_called_once()
        t[0] += 30
        watch.poll(71., .2, lambda: None)
        self.assertTrue(watch.awaiting)
        self.assertIsNone(watch.health()['rgb_reader_last_error'])
        watch.observe(1_000_000_000, .1)
        self.assertFalse(watch.awaiting)

    def test_rebinding_discards_old_queues_callbacks_and_pending_tf_frames(self):
        pending = PendingPair(2., clock=lambda: 0.)
        callbacks, destroyed, seen, signals = [], [], [], []
        def subscribe(cb):
            callbacks.append(cb)
            return cb
        def filters():
            gen = len(callbacks)
            return (lambda msg: signals.append((gen, 'rgb', msg)),
                    lambda msg: signals.append((gen, 'depth', msg)))
        def observe(msg, kind):
            seen.append((kind, msg))
            return msg != 'cached'
        readers = RebindableRGB(subscribe, destroyed.append, filters, observe, pending.clear)
        readers.rebind()
        callbacks[0]('first'); readers.depth('d1')
        pending.offer('old', 'old'); pending.take(); pending.offer('old2', 'old2')
        readers.rebind()
        self.assertIsNone(pending.take())
        callbacks[0]('retired'); callbacks[1]('cached'); callbacks[1]('new'); readers.depth('d2')
        self.assertEqual(destroyed, [callbacks[0]])
        self.assertNotIn(('rgb', 'retired'), seen)
        self.assertEqual(signals, [(0,'rgb','first'),(0,'depth','d1'),(1,'rgb','new'),(1,'depth','d2')])
        readers.close(); readers.close()
        callbacks[1]('late'); readers.depth('d3')
        self.assertEqual(destroyed, callbacks)
        self.assertNotIn(('rgb', 'late'), seen)
        self.assertEqual(len(signals), 4)
