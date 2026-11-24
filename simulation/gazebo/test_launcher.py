"""No Gazebo, ROS sockets or real process signals: launcher lifecycle regressions."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run
import smoke_check


class LauncherTests(unittest.TestCase):
    def test_environment_is_loopback_only_and_removes_inherited_discovery(self):
        inherited = {'CYCLONEDDS_URI': 'robot discovery', 'ROS_DOMAIN_ID': '0',
                     'ROS_DISCOVERY_SERVER': 'robot:11811', 'ROS_STATIC_PEERS': 'robot',
                     'GZ_PARTITION': 'hardware', 'PYTHONPATH': 'incompatible',
                     'DYLD_LIBRARY_PATH': 'conda', 'HOME': '/unchanged', 'PATH': '/bad'}
        with patch.dict(os.environ, inherited, clear=True):
            native, ros = run.environments('/ros/bin/python', Path('/tmp/sim-logs'))
            for env in (native, ros):
                self.assertEqual(env['ROS_LOCALHOST_ONLY'], '1')
                self.assertEqual(env['ROS_AUTOMATIC_DISCOVERY_RANGE'], 'LOCALHOST')
                self.assertEqual(env['GZ_IP'], '127.0.0.1')
                self.assertTrue(env['GZ_PARTITION'].startswith('crossmaps_sim_'))
                self.assertEqual(env['HOME'], '/unchanged')
                for key in ('ROS_DOMAIN_ID', 'CYCLONEDDS_URI', 'ROS_DISCOVERY_SERVER', 'ROS_STATIC_PEERS', 'PYTHONPATH', 'DYLD_LIBRARY_PATH'):
                    self.assertNotIn(key, env)
            self.assertTrue(ros['PATH'].startswith('/ros/bin:'))
            run.apply_environment(ros)
            self.assertNotIn('CYCLONEDDS_URI', os.environ)
            self.assertEqual(dict(os.environ), ros)

    def test_packaged_config_has_fixed_simulation_boundaries(self):
        config = run.validate_simulation_config()
        mutations = [
            lambda c: c['robots'][0].update(domain=0),
            lambda c: c['robots'][1].update(id='rover_a'),
            lambda c: c.update(central_domain=42),
            lambda c: c['dashboard'].update(port=8766),
            lambda c: c['dashboard'].update(host='0.0.0.0'),
            lambda c: c['input'].update(use_sim_time=False),
            lambda c: c['robots'][0].update(camera_relay={'mode': 'auto'}),
            lambda c: c['robots'][1].update(rgb_compressed=False),
            lambda c: c['robots'][1].update(depth_compressed=False),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            for mutate in mutations:
                altered = deepcopy(config)
                mutate(altered)
                path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(RuntimeError, 'refusing to launch'):
                    run.validate_simulation_config(path)

    def test_dead_group_leader_does_not_hide_live_native_helper(self):
        process = Mock(pid=500, poll=Mock(return_value=0), wait=Mock(return_value=0))
        groups = {500}
        def killpg(pid, sig):
            if pid not in groups:
                raise ProcessLookupError()
            if sig:
                groups.remove(pid)
        with patch.object(run.os, 'killpg', side_effect=killpg) as kill:
            run.stop_owned([('bridge', process)], grace=0.)
        kill.assert_any_call(500, signal.SIGINT)
        process.wait.assert_called_once_with(timeout=.1)

    def test_cleanup_handles_exit_race_and_escalates_stubborn_groups(self):
        processes = [Mock(pid=501, poll=Mock(return_value=None), wait=Mock(return_value=0)),
                     Mock(pid=502, poll=Mock(return_value=0), wait=Mock(return_value=0))]
        groups = {501}
        def killpg(pid, sig):
            if pid not in groups:
                raise ProcessLookupError()
            if sig == signal.SIGKILL:
                groups.remove(pid)
        clock = iter(range(100))
        with patch.object(run.os, 'killpg', side_effect=killpg) as kill, \
                patch.object(run.time, 'monotonic', side_effect=lambda: next(clock)), patch.object(run.time, 'sleep'):
            run.stop_owned([('stubborn', processes[0]), ('exited', processes[1])], grace=0.)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            kill.assert_any_call(501, sig)
        self.assertFalse(groups)

    def test_setup_timeout_cleans_entire_owned_process_group(self):
        process = Mock()
        process.wait.side_effect = subprocess.TimeoutExpired(['cmake'], 5.)
        with patch.object(run.subprocess, 'Popen', return_value=process) as spawn, patch.object(run, 'stop_owned') as stop:
            with self.assertRaises(subprocess.TimeoutExpired):
                run.run_bounded(['cmake'], {'isolated': 'yes'}, 5.)
        spawn.assert_called_once_with(['cmake'], env={'isolated': 'yes'}, start_new_session=True)
        stop.assert_called_once_with([('cmake', process)], grace=1.)

    def test_inaccessible_group_does_not_abort_cleanup_of_other_owned_children(self):
        processes = [Mock(pid=801, poll=Mock(return_value=0), wait=Mock(return_value=0)),
                     Mock(pid=802, poll=Mock(return_value=0), wait=Mock(return_value=0))]
        live = {802}
        def signal_group(pid, sig):
            if pid == 801:
                raise PermissionError('protected helper')
            if pid not in live:
                raise ProcessLookupError()
            if sig:
                live.remove(pid)
        with patch.object(run.os, 'killpg', side_effect=signal_group) as kill, patch('sys.stderr') as errors:
            run.stop_owned([('protected', processes[0]), ('other', processes[1])], grace=0.)
        kill.assert_any_call(802, signal.SIGINT)
        self.assertFalse(live)
        self.assertTrue(errors.write.called)

    def test_ros_python_import_probe_has_finite_timeout(self):
        with patch.dict(os.environ, {'CROSSMAPS_ROS_PYTHON': sys.executable}), \
                patch.object(run.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['python'], 20.)) as probe:
            with self.assertRaisesRegex(RuntimeError, 'ROS Python not found'):
                run.ros_python()
        self.assertEqual(probe.call_args.kwargs['timeout'], 20)

    def test_ready_file_must_be_fresh_and_belong_to_current_bridge(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ready.json'
            health = dict(ready=True, robot='rover_a', domain=80, pid=10, updated_unix=100.,
                          advancing_age_s=dict(clock=.2, odom=.1, rgb=.3, depth=.3, scan=.1))
            with patch.object(run.time, 'time', return_value=101.):
                self.assertFalse(run.source_ready(path, 'rover_a', 80, 10))
                path.write_text(json.dumps(health))
                self.assertTrue(run.source_ready(path, 'rover_a', 80, 10))
                self.assertFalse(run.source_ready(path, 'rover_a', 80, 11))
                for change in ({'ready': False}, {'updated_unix': 20.}, {'domain': 0}, {'updated_unix': float('nan')},
                               {'advancing_age_s': dict(clock=.1, odom=.2, rgb=6., depth=.2)}):
                    path.write_text(json.dumps(dict(health, **change)))
                    self.assertFalse(run.source_ready(path, 'rover_a', 80, 10))

    def launch_fixture(self, ready=True, sensors_only=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / '.build').mkdir()
        (root / '.build/gz_capture').write_bytes(b'placeholder')
        processes = [Mock(pid=pid, poll=Mock(return_value=None)) for pid in range(100, 108)]
        patches = [patch.object(run, 'HERE', root),
                   patch.dict(sys.modules, {'generate_world': SimpleNamespace(build=Mock(return_value=root/'world.sdf'))}),
                   patch.object(run.subprocess, 'Popen', side_effect=processes),
                   patch.object(run, 'source_ready', return_value=ready),
                   patch.object(run, 'stop_owned'), patch.object(run.socket, 'socket')]
        mocks = [item.start() for item in patches]
        for item in patches:
            self.addCleanup(item.stop)
        args = SimpleNamespace(sensors_only=sensors_only, skip_build=True, no_gui=True,
                               render_engine='ogre2', duration=.01, sensor_profile='clean', noise_seed=2026, camera_width=960,
                               nav2=False, rviz='none')
        return args, root, processes, mocks

    def test_launch_isolates_domains_and_cleans_on_user_interrupt_during_readiness(self):
        args, root, processes, mocks = self.launch_fixture(ready=False)
        def interrupt(_):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        before = signal.getsignal(signal.SIGINT)
        with patch.object(run.time, 'sleep', side_effect=interrupt):
            self.assertEqual(run.launch(args, '/ros/python', root/'logs', {'native': 'yes'}, {'ros': 'yes'}, '/gz', '/cmake'), 0)
        commands = [call.args[0] for call in mocks[2].call_args_list]
        self.assertEqual(len(commands), 3)  # no coordinator on incomplete sensors
        self.assertIn('80', commands[1]); self.assertIn('81', commands[2])
        self.assertNotIn('0', commands[1]); self.assertNotIn('1', commands[2])
        self.assertTrue(all(call.kwargs['start_new_session'] for call in mocks[2].call_args_list))
        self.assertEqual([name for name, _ in mocks[4].call_args.args[0]], ['gazebo', 'rover_a', 'rover_b'])
        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_ready_launch_uses_only_simulation_config_then_cleans_on_duration(self):
        args, root, processes, mocks = self.launch_fixture()
        clock = iter([0., 1., 2.])
        with patch.object(run.time, 'monotonic', side_effect=lambda: next(clock)):
            self.assertEqual(run.launch(args, '/ros/python', root/'logs', {}, {}, '/gz', '/cmake'), 0)
        coordinator = mocks[2].call_args_list[-1].args[0]
        self.assertIn(str(run.CONFIG), coordinator)
        self.assertNotIn('--resume', coordinator)
        self.assertEqual(len(mocks[4].call_args.args[0]), 5)
        self.assertIn('--simulation-truth', coordinator)
        self.assertIn('--simulation-scene', coordinator)
        ground_truth = mocks[2].call_args_list[-2].args[0]
        self.assertIn(str(root/'ground_truth.py'), ground_truth)
        self.assertIn(str(root/'logs/truth.json'), ground_truth)
        probe = mocks[5].return_value.__enter__.return_value
        probe.setsockopt.assert_called_once_with(run.socket.SOL_SOCKET, run.socket.SO_REUSEADDR, 1)
        probe.bind.assert_called_once_with(('127.0.0.1', 8767))

    def test_smoke_duration_rejects_nan_infinity_zero_before_importing_ros(self):
        for value in ('nan', 'inf', '0', '-1', '601'):
            with self.subTest(value=value), patch.object(sys, 'argv', ['smoke_check.py', '--duration', value]), \
                    patch.object(smoke_check, 'ros_python') as python, patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    smoke_check.main()
                self.assertEqual(error.exception.code, 2)
                python.assert_not_called()

    def test_sensors_only_has_no_evaluator_or_coordinator(self):
        args, root, processes, mocks = self.launch_fixture(sensors_only=True)
        clock = iter([0., 1., 2.])
        with patch.object(run.time, 'monotonic', side_effect=lambda: next(clock)):
            self.assertEqual(run.launch(args, '/ros/python', root/'logs', {}, {}, '/gz', '/cmake'), 0)
        commands = [call.args[0] for call in mocks[2].call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertFalse(any('ground_truth.py' in str(command) or 'crossmaps_collaborative.py' in str(command)
                             for command in commands))

    def test_nav2_both_robots_have_separate_owned_supervisors_and_selected_rviz(self):
        args, root, processes, mocks = self.launch_fixture()
        args.nav2, args.rviz = True, 'rover_b'
        clock = iter([0., 1., 2., 3.])
        with patch.object(run.time, 'monotonic', side_effect=lambda: next(clock)), \
                patch.object(run, 'navigation_ready', return_value=True):
            self.assertEqual(run.launch(args, '/ros/python', root/'logs', {}, {}, '/gz', '/cmake'), 0)
        commands = [call.args[0] for call in mocks[2].call_args_list]
        self.assertEqual(len(commands), 7)
        self.assertIn('rover_a', commands[5])
        self.assertIn('--no-rviz', commands[5])
        self.assertIn('rover_b', commands[6])
        self.assertNotIn('--no-rviz', commands[6])
        self.assertEqual(len(mocks[4].call_args.args[0]), 7)

    def test_navigation_ready_requires_current_owned_process_and_recent_report(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(run.time, 'time', return_value=100.):
            path = Path(folder) / 'nav.json'
            health = dict(ready=True, robot='rover_a', domain=80, pid=12, updated_unix=99.)
            path.write_text(json.dumps(health))
            self.assertTrue(run.navigation_ready(path, 'rover_a', 80, 12))
            self.assertFalse(run.navigation_ready(path, 'rover_a', 80, 13))
            path.write_text(json.dumps(dict(health, updated_unix=70.)))
            self.assertFalse(run.navigation_ready(path, 'rover_a', 80, 12))


if __name__ == '__main__':
    unittest.main()
