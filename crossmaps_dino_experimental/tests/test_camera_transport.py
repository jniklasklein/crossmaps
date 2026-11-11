import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import cv2
import numpy as np
from test_node_identity_integration import make_node, NAMESPACE, DEFAULTS, NODE_PATH


class Compressed:
    def __init__(self, data, fmt):
        self.data, self.format = data, fmt


class CameraTransportTests(unittest.TestCase):
    def test_esam_default_weights_do_not_depend_on_launch_directory(self):
        for variant, dim, heads in (('vitt',192,3),('vits',384,6)):
            node = make_node(); node.params['esam_variant'] = variant; node.device = 'mps'
            builder = Mock(return_value=Mock())
            with patch.dict(sys.modules, {'efficient_sam':SimpleNamespace(),
                    'efficient_sam.efficient_sam':SimpleNamespace(build_efficient_sam=builder)}), \
                    patch.dict(NAMESPACE, __file__=str(NODE_PATH)):
                node.init_esam()
            builder.assert_called_once_with(encoder_patch_embed_dim=dim, encoder_num_heads=heads,
                checkpoint=str(NODE_PATH.parent.parent / 'weights' / f'efficient_sam_{variant}.pt'))
            self.assertIsNotNone(node.esam_model)

    def test_default_topics_match_message_types(self):
        self.assertTrue(DEFAULTS['rgb_compressed'])
        self.assertTrue(DEFAULTS['depth_compressed'])
        self.assertEqual(DEFAULTS['rgb_topic'], '/oak/rgb/image_rect/compressed')
        self.assertEqual(DEFAULTS['depth_topic'], '/oak/stereo/image_raw/compressedDepth')

    def test_depth_png_with_transport_header_preserves_metric_geometry(self):
        depth = np.array([[0, 1200, 3500], [65535, 200, 4500]], dtype=np.uint16)
        ok, png = cv2.imencode('.png', depth)
        self.assertTrue(ok)
        with patch.dict(NAMESPACE, CompressedImage=Compressed):
            for header in (b'', bytes(12), bytes(16)):
                result = make_node().decode_depth_msg(Compressed(header+png.tobytes(), '16UC1; compressedDepth png'))
                np.testing.assert_allclose(result, depth.astype(np.float32)*.001)

    def test_inverse_depth_cannot_silently_be_treated_as_millimeters(self):
        with patch.dict(NAMESPACE, CompressedImage=Compressed):
            with self.assertRaisesRegex(RuntimeError, '32FC1'):
                make_node().decode_depth_msg(Compressed(b'', '32FC1; compressedDepth png'))

    def test_rgb_jpeg_decodes_to_rgb_and_preserves_size(self):
        bgr = np.full((16, 20, 3), (0,0,200), dtype=np.uint8)
        _, data = cv2.imencode('.jpg', bgr)
        with patch.dict(NAMESPACE, CompressedImage=Compressed):
            rgb = make_node().decode_rgb_msg(Compressed(data.tobytes(), 'bgr8; jpeg compressed bgr8'))
        self.assertEqual(rgb.shape, bgr.shape)
        self.assertGreater(int(rgb[0,0,0]), 180)
        self.assertLess(int(rgb[0,0,2]), 10)

    def test_camera_info_remains_observable_without_repeated_log_spam(self):
        node = make_node(); logs=[]
        node.get_logger = lambda:SimpleNamespace(info=logs.append)
        node.destroy_subscription = lambda *_:self.fail('must retain camera health subscription')
        msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1,nanosec=0)),
            k=[500.,0.,320.,0.,501.,240.,0.,0.,1.])
        node.on_cam_info(msg);node.on_cam_info(msg)
        self.assertEqual(len(logs),1)
        self.assertEqual(node.pipeline_health.snapshot()['inputs']['camera_info']['count'],2)
        self.assertEqual((node.fx,node.fy,node.cx,node.cy),(500.,501.,320.,240.))
