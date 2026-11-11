import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pipeline_runtime import LatestFrame, PipelineHealth
from test_node_identity_integration import make_node, NAMESPACE


class RuntimeTests(unittest.TestCase):
    def test_one_frame_is_immediately_eligible_not_every_tenth(self):
        box = LatestFrame()
        box.offer('rgb', 'depth', 1.)
        self.assertEqual(box.take(1., 2., 2.), ('rgb','depth',1.))
        self.assertIsNone(box.take(2.,2.,2.))  # Never repeat stale evidence.

    def test_busy_worker_has_only_one_latest_pending_frame(self):
        box = LatestFrame()
        for i in range(100):
            box.offer(i, i, i*.01)
            self.assertIsNone(box.take(i*.01, 2., 2., busy=True))
        self.assertEqual(box.replaced,99)
        self.assertEqual(box.take(1.,2.,2.)[:2], (99,99))

    def test_rate_limit_and_age_expiry(self):
        box=LatestFrame()
        box.offer(1,1,0.); box.take(0.,2.,2.)
        box.offer(2,2,.1)
        self.assertIsNone(box.take(.1,2.,2.))
        self.assertEqual(box.take(.5,2.,2.)[0],2)
        box.offer(3,3,1.)
        self.assertIsNone(box.take(4.,2.,2.,busy=True))
        self.assertEqual(box.expired,1)

    def test_health_distinguishes_missing_input_sync_and_processing(self):
        now=[0.]; health=PipelineHealth(lambda:now[0])
        self.assertEqual(health.snapshot()['state'],'waiting_for_rgb_and_depth')
        health.input('rgb',1); health.input('depth',1)
        self.assertEqual(health.snapshot()['state'],'waiting_for_synchronization')
        health.input('sync'); health.working('DINO',5)
        self.assertEqual(health.snapshot()['state'],'processing')
        now[0]=31.
        self.assertEqual(health.snapshot()['state'],'processing_stalled')
        health.input('rgb',2); health.input('depth',2); health.input('sync'); health.complete()
        self.assertEqual(health.snapshot()['state'],'healthy')

    def test_repeated_sensor_stamps_are_not_fresh_data(self):
        now=[0.]; health=PipelineHealth(lambda:now[0])
        for i in range(15):
            now[0]=float(i);health.input('rgb',1);health.input('depth',i)
        self.assertEqual(health.snapshot()['state'],'waiting_for_rgb')
        self.assertEqual(health.snapshot()['inputs']['rgb']['age_s'],0.)

    def test_failure_retains_detail_and_success_clears_it(self):
        health=PipelineHealth()
        health.input('rgb'); health.input('depth'); health.input('sync')
        health.idle('model error');health.reject('TF missing')
        self.assertEqual(health.snapshot()['state'],'processing_error')
        self.assertEqual(health.snapshot()['last_rejection'],'TF missing')
        health.complete()
        self.assertIsNone(health.snapshot()['last_error'])

    def test_input_rate_is_not_capped_by_bounded_history(self):
        now=[0.]; health=PipelineHealth(lambda:now[0])
        for i in range(600):
            now[0]=i/30.; health.input('rgb',i)
        self.assertAlmostEqual(health.snapshot()['inputs']['rgb']['hz'],30.)

    def test_real_callback_defers_work_until_timer_and_drops_obsolete_pairs(self):
        node=make_node(); calls=[]
        node._process_rgb_depth=lambda a,b:calls.append((a,b))
        node.on_rgb_depth('a','a');node.on_rgb_depth('b','b')
        self.assertFalse(calls)
        node.on_observation_timer()
        self.assertEqual(calls,[('b','b')])
        node.on_observation_timer()
        self.assertEqual(len(calls),1)

    def test_real_timer_does_not_preprocess_while_worker_busy(self):
        node=make_node()
        node._process_rgb_depth=lambda *_:self.fail('worker already busy')
        node.async_worker_busy.set()
        node.on_rgb_depth('a','b');node.on_observation_timer()
        self.assertIsNotNone(node.latest_frame.pending)

    def test_failed_preprocessing_is_reported_not_a_ros_executor_crash(self):
        node=make_node()
        node._process_rgb_depth=lambda *_:(_ for _ in ()).throw(ValueError('bad frame'))
        node.on_rgb_depth('a','b');node.on_observation_timer()
        self.assertIn('bad frame',node.pipeline_health.snapshot()['last_error'])
        self.assertIsNone(node.latest_frame.pending)

    def test_failed_fusion_is_reported_and_does_not_repeat_result(self):
        node=make_node()
        node.apply_semantic_result=lambda *_:(_ for _ in ()).throw(ValueError('bad fusion'))
        node.semantic_result_queue.put(dict(seq=1))
        node.apply_finished_semantic_jobs()
        self.assertIn('bad fusion',node.pipeline_health.snapshot()['last_error'])
        self.assertTrue(node.semantic_result_queue.empty())

    def test_publication_budget_leaves_callback_time_and_does_not_backlog(self):
        node=make_node(); now=[1.]; calls=[]
        def publish():
            calls.append(now[0]);now[0]+=.35
        node._publish_cycle=publish
        with patch.object(NAMESPACE['time'],'perf_counter',side_effect=lambda:now[0]):
            node.on_publish()
            now[0]=1.5;node.on_publish()
            self.assertEqual(len(calls),1)
            now[0]=2.1;node.on_publish()
            self.assertEqual(len(calls),2)
        self.assertEqual(node.publish_skipped_ticks,1)


if __name__=='__main__': unittest.main()
