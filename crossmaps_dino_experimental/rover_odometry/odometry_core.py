"""Wheel-position odometry with an independent physical gyro consistency check.

All times supplied by the caller are monotonic seconds, except optional IMU
acquisition_s (used only for continuity/queue checks). Distances in metres,
angles in radians. Never feed RF2O's synthesized /imu/data into this estimator.
The physical gyro never integrates a second heading. No ROS, TF, wall clock,
fixed-pose setting or motion-command dependency.
"""
from collections import deque
from dataclasses import dataclass
import math
from statistics import median


@dataclass(frozen=True)
class Config:
    wheel_base: float = .175
    wheel_step: float = .01                  # observed vendor wheel units: cm / 100
    max_wheel_speed: float = 2.0             # plausibility guard, not a motion command
    wheel_gap: float = .65
    gyro_gap: float = .24
    gyro_sign: int = 1
    use_gyro: bool = True
    bias_window: float = 2.5
    quiet_gyro: float = .035                 # > observed parked noise on both units
    bias_limit: float = .045

    def __post_init__(self):
        for name in ("wheel_base", "wheel_step", "max_wheel_speed", "wheel_gap", "gyro_gap",
                     "bias_window", "quiet_gyro", "bias_limit"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.gyro_sign not in (-1, 1):
            raise ValueError("invalid gyro sign")


@dataclass(frozen=True)
class Sample:
    x: float
    y: float
    yaw: float
    v: float
    w: float
    pose_cov: tuple
    twist_cov: tuple
    source: str


def _wrap(yaw):
    return math.atan2(math.sin(yaw), math.cos(yaw))


def _cov36(planar, z_var, tilt_var):
    result = [0.] * 36
    for i, a in enumerate((0, 1, 5)):
        for j, b in enumerate((0, 1, 5)):
            result[6*a+b] = planar[i][j]
    result[14], result[21], result[28] = z_var, tilt_var, tilt_var
    return tuple(result)


class Estimator:
    def __init__(self, config=None, initial_pose=(0., 0., 0.)):
        self.cfg = config if config is not None else Config()
        self.x, self.y, self.yaw = map(float, initial_pose)
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise ValueError("initial_pose must be finite")
        self.yaw = _wrap(self.yaw)
        # Local starting reference, not an absolute map localization. Positive
        # uncertainty floors avoid both near-perfect constraints and singular axes.
        self.P = [[.05**2, 0., 0.], [0., .05**2, 0.], [0., 0., .08**2]]
        self.raw = deque(maxlen=1024)  # (receive time, physical body Z rate)
        self.last_g = self.last_stamp = self.offset = None
        self.base = self.pending = None
        self.quiet_since = self.motion_since = None
        self.unobserved_angle = 0.
        self.next_bias = self.bias = None
        self.verified = False   # corroboration diagnostic only
        self.suspect = False
        self.window_t = None
        self.window_w = self.window_g = self.agree = self.oppose = 0.
        self.vhist = deque()
        self.status = "waiting_wheels"
        self.clock_resets = self.wheel_resets = self.rejected_gyro = self.gaps = 0

    def gyro(self, t_monotonic, wz_rad_s, acquisition_s=None):
        c = self.cfg
        t, g = t_monotonic, wz_rad_s
        if not c.use_gyro:
            return
        if not math.isfinite(t) or not math.isfinite(g) or abs(g) > 8.:
            self.rejected_gyro += 1
            return
        if self.last_g is not None and t <= self.last_g:
            self.rejected_gyro += 1
            return
        if acquisition_s is not None:
            if not math.isfinite(acquisition_s) or acquisition_s <= 0:
                self.rejected_gyro += 1
                return
            if self.last_stamp is not None:
                if acquisition_s == self.last_stamp:
                    self.rejected_gyro += 1
                    return
                if acquisition_s < self.last_stamp:
                    # One reordered packet must not activate a stale stream.
                    if self.last_stamp-acquisition_s < .5:
                        self.rejected_gyro += 1
                        return
                    self.raw.clear()
                    self.offset = None
                    self.clock_resets += 1
                elif self.last_g is not None and acquisition_s-self.last_stamp > t-self.last_g + 2.:
                    # Rover wall time stepped forward. Keep the wheel map origin,
                    # but never integrate across an IMU acquisition clock step.
                    self.raw.clear()
                    self.offset = None
                    self.clock_resets += 1
            self.last_stamp = acquisition_s
            off = t - acquisition_s
            # Local board stamps; absolute epoch offset is irrelevant. Reject
            # queue bursts older than our best observed relative transport time.
            self.offset = off if self.offset is None else min(off, self.offset)
            if off - self.offset > c.gyro_gap:
                self.rejected_gyro += 1
                return
        if self.last_g is not None and t-self.last_g > c.gyro_gap:
            self.raw.clear()
            self.quiet_since = None
            self._window_reset()
        self.raw.append((t, g))
        self.last_g = t

    def _window_reset(self):
        self.window_t = None
        self.window_w = self.window_g = 0.

    def _calibrate(self, t, still, fresh):
        c = self.cfg
        if not still or not fresh:
            self.quiet_since = None
            return
        if self.quiet_since is None:
            self.quiet_since = t
        if t-self.quiet_since < c.bias_window or (self.next_bias is not None and t < self.next_bias):
            return
        recent = [(ts, g) for ts, g in self.raw if t-c.bias_window <= ts <= t]
        if len(recent) < 12 or recent[-1][0] - recent[0][0] < .80*c.bias_window:
            return
        self.next_bias = t + 1.
        values = [g for _, g in recent]
        center = median(values)
        mad = median([abs(g-center) for g in values])
        if abs(center) > c.bias_limit or mad > .017 or max(values) - min(values) > .08:
            return
        if self.bias is None:
            self.bias = center
        elif abs(center-self.bias) <= .015:
            self.bias += .05 * (center-self.bias)

    def _mean_rate(self, begin, end):
        # Piecewise-constant receipt-time integration. Interpolate no more than
        # gyro_gap at an edge and never use readings from after 'end'.
        # Don't extend the old turn rate across entire wheel intervals when the
        # gyro stops (particularly dangerous at a direction change). gyro_gap
        # defines reception health, not a license to integrate a held value.
        if self.last_g is None or self.last_g <= begin or end-self.last_g > self.cfg.gyro_gap:
            return None
        buf = []
        for ts, g in reversed(self.raw):
            if ts > end:
                continue
            buf.append((ts, g))
            if ts <= begin:
                break
        if not buf or buf[-1][0] > begin or begin-buf[-1][0] > self.cfg.gyro_gap:
            return None
        pos, total = end, 0.
        for ts, g in buf:
            start = max(begin, ts)
            total += g*(pos-start)
            pos = start
        return self.cfg.gyro_sign * (total/(end-begin) - (self.bias or 0.))

    def _corroborate(self, t, dw, dg):
        # Encoder quantization is ~0.057 rad per one-wheel 1 cm step. Check
        # direction and magnitude over windows, not instantaneous wheel rates.
        if self.window_t is None:
            self.window_t = t
        self.window_w += dw
        self.window_g += dg
        if t-self.window_t >= .6:
            w, g = self.window_w, self.window_g
            if max(abs(w),abs(g)) >= .09:
                if abs(w) >= .07 and w*g > 0 and .55 <= abs(g/w) <= 1.8:
                    self.agree += min(abs(w), abs(g), .17)
                    self.oppose = max(0., self.oppose-.06)
                    if self.agree >= .34 and self.oppose < .05:
                        self.verified = True
                        self.suspect = False
                else:
                    self.oppose += min(abs(w-g), .20)
                    self.agree = max(0., self.agree-.17)
                    if self.oppose >= .25:
                        self.suspect = True
                        self.verified = False
            self._window_reset()

    def _inflate_gap(self, dt):
        # Unknown displacement during dropped/reset measurements. Never make up
        # a trajectory across an outage or retain the old certainty.
        d = min(max(dt, 0.), 30.)
        self.P[0][0] += (.05 + .2*d)**2
        self.P[1][1] += (.05 + .2*d)**2
        self.P[2][2] += (.08 + .4*d)**2
        self.vhist.clear()
        self._window_reset()
        self.quiet_since = self.motion_since = None
        self.unobserved_angle = 0.

    def _rates(self, begin, end, distance, angle):
        self.vhist.append((begin, end, distance, angle))
        while len(self.vhist)>1 and self.vhist[0][1] <= end-.25:
            self.vhist.popleft()
        span = end-self.vhist[0][0]
        return sum(x[2] for x in self.vhist)/span, sum(x[3] for x in self.vhist)/span, span

    def wheel(self, t_monotonic, left_m, right_m):
        t, l, r = t_monotonic, left_m, right_m
        if not all(math.isfinite(v) for v in (t, l, r)):
            self.status = "invalid_wheels"
            return None
        if self.base is None:
            self.base = (t, l, r)
            self.status = "wheel_baseline"
            return None
        prev, old_l, old_r = self.base
        dt = t-prev
        c = self.cfg
        if dt <= 0.:
            self.status = "out_of_order_wheels"
            return None
        if dt < .005:
            # Unstamped queued wheel reports can be delivered in a burst.
            # Keep the previous baseline, accumulating displacement until the
            # receive interval supports a meaningful finite velocity.
            self.status = "coalescing_wheels"
            return None
        dl, dr = l-old_l, r-old_r
        if dt > c.wheel_gap:
            self._inflate_gap(dt)
            self.base, self.pending = (t,l,r), None
            self.gaps += 1
            self.status = "wheel_gap"
            return None
        def plausible(a, b, elapsed):
            return max(abs(a), abs(b)) <= c.max_wheel_speed*elapsed + 1.2*c.wheel_step
        if not plausible(dl, dr, dt):
            p = self.pending
            if p and 0 < t-p[0] <= c.wheel_gap and plausible(l-p[1], r-p[2], t-p[0]):
                self._inflate_gap(dt)
                self.base, self.pending = (t,l,r), None
                self.wheel_resets += 1
                self.status = "wheel_reset"
            else:
                self.pending = (t,l,r)
                self.status = "wheel_spike"
            return None
        self.base, self.pending = (t,l,r), None
        still = abs(dl) + abs(dr) < 1e-7
        fresh = c.use_gyro and self.last_g is not None and 0 <= t-self.last_g <= c.gyro_gap
        self._calibrate(t, still, fresh)
        g = self._mean_rate(prev, t) if fresh else None
        if g is None and fresh:
            # May use the latest bounded-age rate to CHECK for inconsistency
            # across callback skew. It never contributes to the odometry pose.
            g = c.gyro_sign * (self.raw[-1][1] - (self.bias or 0.))
        ds, dw = (dl+dr)/2., (dr-dl)/c.wheel_base
        if g is not None and self.bias is not None:
            self._corroborate(t, dw, g*dt)
        else:
            self._window_reset()
        # Only wheel displacement is integrated into heading. Even a previously
        # correlated gyro may stick, reverse or measure motion with no traction.
        # Report independent evidence, never double-count later encoder ticks.
        dyaw = dw
        mode = "wheel"
        if still and g is None and self.status == "unverified_rotation":
            # Losing the monitor is not evidence an unresolved rotation ended.
            self._rates(prev,t,0.,0.)
            return None
        if still and g is not None and abs(g) > c.quiet_gyro:
            if self.motion_since is None:
                self.motion_since = prev
            before = self.unobserved_angle
            self.unobserved_angle = min(math.pi, before + abs(g)*dt)
            # Two positions are quantized independently: on a slow pivot even
            # 1.4 s without new wheel ticks can be expected. Require a gyro arc
            # exceeding the physically plausible unreported encoder motion.
            if t-self.motion_since >= .25 and self.unobserved_angle > 2.3*c.wheel_step/c.wheel_base:
                self.status = "unverified_rotation"
                self.P[2][2] += self.unobserved_angle**2 - before**2
                self._rates(prev,t,0.,0.)
                return None
            mode = "wheel_turn_pending"
        else:
            self.motion_since, self.unobserved_angle = None, 0.
            if still:
                mode = "stationary" if g is not None and abs(g) <= c.quiet_gyro else "wheel_only_quiet"
            elif self.suspect:
                mode = "wheel_gyro_disagree"
        # Body-frame midpoint integration with exact arc-length compensation.
        h = .5*dyaw
        arc = ds*(math.sin(h)/h if abs(h) > 1e-7 else 1.)
        co, si = math.cos(self.yaw+h), math.sin(self.yaw+h)
        dx, dy = arc*co, arc*si
        self.x += dx
        self.y += dy
        self.yaw = _wrap(self.yaw+dyaw)
        # Smooth reported rates over a short causal window to avoid turn-rate
        # impulses from individual encoder ticks; pose uses each increment once.
        v, w, span = self._rates(prev,t,ds,dyaw)
        # Explicit dead reckoning / slip process error, not a claimed hardware
        # calibration. Propagate previous orientation uncertainty into x/y.
        qd = (.006*dt + .05*abs(ds))**2
        ql = (.008*dt + .10*abs(ds))**2
        qa = (.018*dt + .14*abs(dyaw) + .05*abs(ds))**2
        degraded = g is None or mode in ("wheel_gyro_disagree", "wheel_turn_pending", "wheel_only_quiet")
        if degraded:
            qa += (.18*dt)**2 + (.35*abs(dyaw))**2
        if mode == "wheel_gyro_disagree":
            qd += (.25*abs(ds))**2
        F = ((1.,0.,-dy), (0.,1.,dx), (0.,0.,1.))
        G = ((co,-si,-.5*dy), (si,co,.5*dx), (0.,0.,1.))
        q = (qd, ql, qa)
        self.P = [[sum(F[i][a]*self.P[a][b]*F[j][b] for a in range(3) for b in range(3))
                   + sum(G[i][k]*q[k]*G[j][k] for k in range(3))
                   for j in range(3)] for i in range(3)]
        sv = .065 + c.wheel_step/(span*math.sqrt(6)) + .10*abs(v)
        sw = .12 + c.wheel_step/(c.wheel_base*span*math.sqrt(3)) + .14*abs(w)
        if degraded:
            sw += .35
        pv = _cov36(self.P, 2., 1.)
        tv = _cov36(((sv**2,0.,0.),(0.,(.10+.15*abs(v))**2,0.),(0.,0.,sw**2)), 2., 1.)
        self.status = mode
        return Sample(self.x, self.y, self.yaw, v, w, pv, tv, mode)

    def health(self, t_monotonic):
        return dict(status=self.status, heading_source="differential_wheels",
                    wheel_age_s=None if self.base is None else t_monotonic-self.base[0],
                    gyro_age_s=None if self.last_g is None else t_monotonic-self.last_g,
                    gyro_bias_rad_s=self.bias, gyro_sign_verified=self.verified, gyro_suspect=self.suspect,
                    gyro_rejections=self.rejected_gyro, gyro_clock_resets=self.clock_resets,
                    wheel_resets=self.wheel_resets, wheel_gaps=self.gaps)
