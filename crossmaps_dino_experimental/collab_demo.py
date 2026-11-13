"""Deterministic textured RGB-D fixture. Not a model or rover accuracy demo."""
import time
import cv2
import numpy as np


def _scene(side):
    h, w = 240, 400
    k = np.array([[290., 0, w/2], [0, 290., h/2], [0, 0, 1]])
    v, u = np.mgrid[:h, :w]
    # Fronto-parallel planes at several depths, ray traced with a moving camera.
    depth = np.full((h, w), 3.4, np.float32)
    labels = np.zeros((h, w), np.uint8)
    for sid, z, x0, x1, y0, y1 in [(1, 2.0, -.85, -.18, -.35, .45),
                                    (2, 2.0, -.18, .62, -.35, .45),
                                    (3, 1.6, .85, 1.2, -.1, .55),
                                    (4, 2.7, -.9, .9, .6, .9)]:
        x, y = (u-k[0, 2])*z/k[0, 0]+side, (v-k[1, 2])*z/k[1, 1]
        mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        depth[mask], labels[mask] = z, sid
    x = (u-k[0, 2])*depth/k[0, 0]+side
    y = (v-k[1, 2])*depth/k[1, 1]
    ix, iy = np.floor(x*85).astype(np.int64), np.floor(y*85).astype(np.int64)
    noise = ((ix*73856093) ^ (iy*19349663)) & 255
    rgb = np.stack(((noise+labels*40) % 255, (noise*3+labels*30) % 255, (noise*7+80) % 255), axis=-1).astype(np.uint8)
    return rgb, depth, k, labels


def observations(step, robots=None):
    robots = robots or [{'id':'rover_a'}, {'id':'rover_b'}]
    output = []
    camera_ros = np.eye(4)
    camera_ros[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    palettes = np.array([[95, 115, 140], [110, 190, 75], [195, 95, 200], [80, 180, 235], [195, 165, 85]], np.uint8)
    for index, robot in enumerate(robots):
        side = .04*min(step, 12) + .03*index
        bgr, depth, k, labels = _scene(side)
        world_camera = camera_ros.copy()
        world_camera[:3, 3] = [0, -side, .6]
        world_local = np.eye(4)
        world_local[0, 3] = 1.5*index
        world_local[1, 3] = .5*index
        local_camera = np.linalg.inv(world_local) @ world_camera
        v, u = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
        z = depth[v, u]
        points = np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), axis=-1).reshape(-1, 3)
        ids = labels[v, u].reshape(-1)
        surfaces = []
        for sid in range(1, 5):
            feature = np.eye(4, dtype=np.float32)[0 if sid in (1, 2) else sid-1]
            surfaces.append(dict(local_id=str(sid), entity_id='sofa' if sid in (1, 2) else f'object{sid}',
                                 points_camera=points[ids == sid], dino=feature, clip=feature, quality=1.))
        seg = cv2.addWeighted(bgr, .3, palettes[labels], .7, 0)
        entity_labels = labels.copy(); entity_labels[labels == 2] = 1
        ent = cv2.addWeighted(bgr, .3, palettes[entity_labels], .7, 0)
        stamp = time.time_ns()
        output.append(dict(robot_id=robot['id'], session_id='synthetic_demo_v1', seq=step+1,
            stamp_ns=stamp, rgb_stamp_ns=stamp, received_monotonic=time.monotonic(),
            calibration_id='synthetic_400_240_v1', feature_space_id='synthetic_descriptor_v1',
            T_local_camera=local_camera, K=k, rgb=bgr, depth=depth, points_camera=points,
            colors=bgr[v, u, ::-1].reshape(-1, 3), global_descriptor=np.array([1., 0., 0., 0.]),
            surfaces=surfaces, pose_quality='synthetic_ground_truth', depth_registration='synthetic_registered',
            mask_count=4, segment_count=4, entity_count=3,
            demo_images={'sam':seg, 'segments':seg, 'entities':ent}))
    return output
