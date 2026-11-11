import unittest
from unittest.mock import Mock, patch
import numpy as np
from test_node_identity_integration import make_node, NAMESPACE


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.node=make_node()
        rng=np.random.default_rng(8)
        self.keys=[(i,0,0) for i in range(2300)]
        self.sums={k:rng.normal(size=64).astype(np.float32) for k in self.keys}
        self.weights={k:float(rng.uniform(.01,30.)) for k in self.keys}
        self.views={k:i%256 for i,k in enumerate(self.keys)}
        self.node.text_emb=rng.normal(size=64).astype(np.float32)
        self.node.text_emb/=np.linalg.norm(self.node.text_emb)

    def test_batched_scoring_matches_scalar_across_chunks_and_flags(self):
        for bins in (1,8):
            self.node.params['view_bins']=bins
            for view,conf,sem in ((False,False,False),(True,True,True),(False,True,False),(True,False,True)):
                args=(self.keys,self.sums,self.weights,self.views,50.,view,conf,sem)
                scalar=self.node._compute_keyed_scores_conf_scalar(*args)
                batch=self.node._compute_keyed_scores_conf(*args)
                self.assertEqual(batch[0],scalar[0])
                for got,want in zip(batch[1:],scalar[1:]):
                    np.testing.assert_allclose(got,want,atol=2e-6,rtol=2e-5)

    def test_empty_invalid_and_missing_descriptors_are_not_nan(self):
        self.sums[self.keys[0]][:]=0.
        self.sums[self.keys[1]][:]=np.nan
        self.weights[self.keys[2]]=0.
        self.node.text_emb=None
        result=self.node._compute_keyed_scores_conf(self.keys[:3],self.sums,self.weights,self.views,50.,True,True,True)
        self.assertIsNone(result)
        self.assertIsNone(self.node._compute_keyed_scores_conf([],{}, {}, {},50.,True,True,True))

    def test_fused_scores_shared_within_callback_not_across_observations(self):
        node=self.node
        node.vox_stm_sum,self.node.vox_stm_w,self.node.vox_stm_viewmask=self.sums,self.weights,self.views
        node._entity_score_context={}
        bundle=node._voxel_memory_bundle('stm_ltm')
        with patch.dict(NAMESPACE, score_voxels=Mock(wraps=NAMESPACE['score_voxels'])):
            scorer=NAMESPACE['score_voxels']
            for keys in (self.keys[:10],self.keys[20:50],self.keys):
                result=node._compute_keyed_scores_conf(keys,*bundle[1:],True,True,True)
                self.assertEqual(result[0],keys)
            self.assertEqual(scorer.call_count,1)
            node._entity_score_context=None
            node._compute_keyed_scores_conf(self.keys[:1],*bundle[1:],True,True,True)
            self.assertEqual(scorer.call_count,2)

    def test_publication_context_is_discarded_even_on_error(self):
        node=self.node
        node.publish_segment_instance_voxels=lambda *_:(_ for _ in ()).throw(ValueError('test'))
        node.publish_segment_query_voxels=node.publish_segment_unassigned_query_voxels=node.publish_object_graph=lambda *_:None
        with self.assertRaises(ValueError):node.publish_entity_views(np.eye(4))
        self.assertIsNone(node._entity_score_context)


if __name__=='__main__':unittest.main()
