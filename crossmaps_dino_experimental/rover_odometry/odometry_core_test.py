"""No ROS required. Exercise physical units, degradation, and estimator observability."""
import math
import random
import unittest

from odometry_core import Config, Estimator


def perform(e, n, t=20., l=0., r=0., v=0., w=0., gyro_rate=None, gyro=True):
    out=[]
    for _ in range(n):
        t += .05
        l += (v-.5*e.cfg.wheel_base*w)*.05
        r += (v+.5*e.cfg.wheel_base*w)*.05
        if gyro:
            e.gyro(t, (w if gyro_rate is None else gyro_rate)-.009, t+1000)
        out.append(e.wheel(t,l,r))
    return t,l,r,out


class OdometryTest(unittest.TestCase):
    def test_even_large_nonzero_encoder_origin_creates_no_motion(self):
        e=Estimator(initial_pose=(12.,-5.,.8))
        t,l,r,rows=perform(e,200,l=405.3,r=-911.5)
        self.assertEqual((rows[-1].x,rows[-1].y,rows[-1].yaw),(12.,-5.,.8))
        self.assertAlmostEqual(e.health(t)['gyro_bias_rad_s'],-.009)
        self.assertFalse(e.health(t)['gyro_sign_verified'])

    def test_observable_motion_from_wheels_and_corroborated_gyro(self):
        e=Estimator()
        t,l,r,_=perform(e,100)
        t,l,r,rows=perform(e,20,t,l,r,v=.2)
        self.assertAlmostEqual(rows[-1].x,.2)
        t,l,r,rows=perform(e,65,t,l,r,w=.45)
        self.assertTrue(e.verified)
        self.assertAlmostEqual(rows[-1].yaw,65*.05*.45,delta=.025)

    def test_strong_unverified_rotation_quarantined_not_published_as_zero(self):
        e=Estimator()
        t,l,r,_=perform(e,100)
        t,l,r,rows=perform(e,30,t,l,r,gyro_rate=.45)
        self.assertTrue(all(s is None for s in rows[10:]))
        self.assertEqual(e.health(t)['status'],'unverified_rotation')

    def test_even_previously_corroborated_gyro_cannot_invent_wheel_quiet_turn(self):
        e=Estimator()
        t,l,r,_=perform(e,100)
        t,l,r,_=perform(e,70,t,l,r,w=.45)
        self.assertTrue(e.verified)
        before=e.yaw
        t,l,r,rows=perform(e,40,t,l,r,gyro_rate=.25)
        self.assertIsNone(rows[-1])
        self.assertAlmostEqual(e.yaw,before)
        t,l,r,rows=perform(e,300,t,l,r)
        self.assertAlmostEqual(e.yaw,before)

    def test_wrong_sign_downgrades_to_wheels(self):
        e=Estimator()
        t,l,r,_=perform(e,100)
        t,l,r,rows=perform(e,100,t,l,r,w=-.4,gyro_rate=.4)
        self.assertTrue(e.suspect)
        self.assertFalse(e.verified)
        self.assertAlmostEqual(rows[-1].yaw,-2.,delta=.01)

    def test_sparse_ticks_cover_arc_once_despite_previously_matching_gyro(self):
        e=Estimator()
        t,l,r,_=perform(e,100)
        t,l,r,_=perform(e,70,t,l,r,w=.45)
        self.assertTrue(e.verified)
        # Align the starting baseline to a real quantized report.
        before=e.yaw; lo=l; ro=r; output=0
        for _ in range(420):
            t+=.05
            l-=.5*e.cfg.wheel_base*.08*.05; r+=.5*e.cfg.wheel_base*.08*.05
            e.gyro(t,.08-.009,t+1000)
            s=e.wheel(t,round(l,2),round(r,2))
            output+=int(s is not None)
        expected = ((round(r,2)-ro)-(round(l,2)-lo))/e.cfg.wheel_base
        self.assertGreater(output,405)
        self.assertAlmostEqual(math.atan2(math.sin(e.yaw-before),math.cos(e.yaw-before)),expected)

    def test_stuck_zero_gyro_in_real_encoder_turn_never_changes_measured_heading(self):
        e=Estimator()
        t,l,r,_=perform(e,100)
        t,l,r,_=perform(e,80,t,l,r,w=.5)
        self.assertTrue(e.verified)
        before=e.yaw
        t,l,r,rows=perform(e,50,t,l,r,w=.45,gyro_rate=0.)
        self.assertTrue(e.suspect)
        self.assertEqual(rows[-1].source,"wheel_gyro_disagree")
        self.assertAlmostEqual(math.atan2(math.sin(e.yaw-before),math.cos(e.yaw-before)),1.125)
        self.assertGreater(rows[-1].twist_cov[35],.30)

    def test_dropout_immediate_wheel_fallback(self):
        e=Estimator()
        t,l,r,_=perform(e,100)
        t,l,r,_=perform(e,20,t,l,r,w=-.4)
        before=e.yaw
        t,l,r,rows=perform(e,20,t,l,r,w=.4,gyro=False)
        self.assertEqual(rows[0].source,"wheel")
        self.assertAlmostEqual(e.yaw-before,.4)

    def test_wheel_reset_spike_out_of_order_and_stale(self):
        e=Estimator()
        t,l,r,rows=perform(e,20)
        p=rows[-1]
        self.assertIsNone(e.wheel(t,100.,100.))
        self.assertIsNone(e.wheel(t+.05,100.,100.))
        self.assertEqual(e.status,"wheel_spike")
        row=e.wheel(t+.10,.01,.01)
        self.assertAlmostEqual(row.x-p.x,.01)
        self.assertIsNone(e.wheel(t+.15,100.,100.))
        self.assertIsNone(e.wheel(t+.20,100.,100.))
        self.assertEqual(e.status,"wheel_reset")
        self.assertIsNone(e.wheel(t+3.,106.,106.))
        self.assertEqual(e.status,"wheel_gap")
        self.assertGreater(e.P[2][2],p.pose_cov[35])
        self.assertAlmostEqual(e.wheel(t+3.05,106.,106.).x,.01)

    def test_gyro_duplicate_stale_clock_reset(self):
        e=Estimator()
        e.gyro(1.,.2,100.)
        e.gyro(1.05,.2,100.)        # republished stamp
        e.gyro(1.6,.2,100.01)       # queued old stamp
        e.gyro(1.7,float('nan'),100.7)
        self.assertEqual(e.last_g,1.)
        self.assertEqual(e.rejected_gyro,3)
        e.gyro(2.,.2,80.)           # source clock restart: rebaseline
        self.assertEqual(e.clock_resets,1)
        self.assertEqual(len(e.raw),1)
        e.gyro(2.05,.2,280.)        # forward clock step
        self.assertEqual(e.clock_resets,2)
        self.assertIsNone(e._mean_rate(2.,2.05))

    def test_invalid_and_high_rate_wheels_do_not_overflow(self):
        e=Estimator()
        self.assertIsNone(e.wheel(1.,0.,0.))
        self.assertIsNone(e.wheel(1.0000001,.009,.009))
        self.assertIsNone(e.wheel(1.01,float('nan'),0.))
        self.assertLess(e.wheel(1.05,.009,.009).v,1.)

    def test_covariances_symmetric_positive_definite_and_nontrivial(self):
        e=Estimator()
        t,l,r,_=perform(e,100)
        t,l,r,rows=perform(e,100,t,l,r,v=.2,w=.2)
        s=rows[-1]
        for c in [s.pose_cov,s.twist_cov]:
            self.assertEqual(len(c),36)
            for i in range(6):
                self.assertGreater(c[7*i],1e-4)
                for j in range(6):
                    self.assertAlmostEqual(c[6*i+j],c[6*j+i])
            rng=random.Random(727)
            for _ in range(50):
                v=[rng.gauss(0,1) for j in range(6)]
                self.assertGreater(sum(v[i]*v[j]*c[6*i+j] for i in range(6) for j in range(6)),0.)
        self.assertNotEqual(s.pose_cov,s.twist_cov)

    def test_configuration_guard(self):
        for args in [dict(gyro_sign=0),dict(wheel_base=0),dict(bias_limit=math.nan)]:
            with self.assertRaises(ValueError):
                Config(**args)


if __name__=='__main__':
    unittest.main()
