"""Supervise only the OAK launch. RTAB-Map remains owned by the original bringup."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def start(context):
    # symlink-install resolves back into the vendor source folder.
    script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crossmaps_oak_guard.py')
    cmd = ['python3', '-u', script, '--owner-pid', str(os.getpid())]
    for arg in ['name', 'params_file', 'rectify_rgb']:
        cmd.extend(['--' + arg.replace('_', '-'), LaunchConfiguration(arg).perform(context)])
    return [ExecuteProcess(cmd=cmd, output='screen', sigterm_timeout='15', sigkill_timeout='8')]


def generate_launch_description():
    default_config = os.path.join(get_package_share_directory('ugv_vision'), 'config', 'oak_d_lite.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('name', default_value='oak'),
        DeclareLaunchArgument('rectify_rgb', default_value='False'),
        DeclareLaunchArgument('params_file', default_value=default_config),
        OpaqueFunction(function=start),
    ])
