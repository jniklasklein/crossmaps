import unittest
from unittest.mock import Mock

from collab_transport import PendingPair, RGBReaderWatch, RebindableRGB


class RGBReaderRecoveryTests(unittest.TestCase):
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
