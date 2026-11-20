"""Check the shared loader without weights, a GPU, or downloading anything."""
import sys
import unittest
from unittest.mock import MagicMock, patch

from clip_inference import load_clip_for_inference


class ClipInferenceTests(unittest.TestCase):
    def setUp(self):
        self.clip, self.timm = MagicMock(), MagicMock()
        self.model = MagicMock()
        self.model.eval.return_value = self.model.to.return_value = self.model
        self.clip.create_model_and_transforms.return_value = self.model, 'train transform', 'eval transform'
        self.clip.get_tokenizer.return_value = 'native tokenizer'
        self.clip.list_pretrained_tags_by_model.return_value = ['dfndr2b']
        self.timm.reparameterize_model.return_value = self.model
        modules = patch.dict(sys.modules, {'open_clip': self.clip, 'timm.utils': self.timm})
        modules.start()
        self.addCleanup(modules.stop)

    def test_mobile_uses_registered_eval_transform_and_fuses_before_device_transfer(self):
        def fuse(model, inplace):
            self.assertTrue(inplace)
            model.eval.assert_called_once()
            model.to.assert_not_called()
            return model
        self.timm.reparameterize_model.side_effect = fuse
        model, preprocess, tokenizer = load_clip_for_inference('MobileCLIP2-S2', 'dfndr2b', 'mps')
        self.assertEqual((model, preprocess, tokenizer), (self.model, 'eval transform', 'native tokenizer'))
        self.clip.create_model_and_transforms.assert_called_once_with(model_name='MobileCLIP2-S2', pretrained='dfndr2b')
        self.clip.get_tokenizer.assert_called_once_with('MobileCLIP2-S2')
        self.timm.reparameterize_model.assert_called_once_with(self.model, inplace=True)
        self.model.to.assert_called_once_with('mps')

    def test_legacy_model_loads_without_fastvit_fusion(self):
        load_clip_for_inference('ViT-L-14', 'openai', 'cpu')
        self.timm.reparameterize_model.assert_not_called()
        self.clip.create_model_and_transforms.assert_called_once_with(model_name='ViT-L-14', pretrained='openai')

    def test_incompatible_tag_or_registry_never_silently_loads_different_weights(self):
        with self.assertRaisesRegex(ValueError, 'dfndr2b'):
            load_clip_for_inference('MobileCLIP2-S2', 'openai', 'cpu')
        self.clip.list_pretrained_tags_by_model.return_value = []
        with self.assertRaisesRegex(RuntimeError, '3.2'):
            load_clip_for_inference('MobileCLIP2-S2', 'dfndr2b', 'cpu')
        self.clip.create_model_and_transforms.assert_not_called()
