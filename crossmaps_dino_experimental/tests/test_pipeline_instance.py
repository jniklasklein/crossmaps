import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from pipeline_instance import PipelineInstance, check_existing_publishers


class InstanceTests(unittest.TestCase):
    def test_duplicate_rejected_and_release_allows_restart(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'pipeline.lock'
            with PipelineInstance(path):
                with self.assertRaisesRegex(RuntimeError,'already owns'):
                    with PipelineInstance(path):pass
            # Persistent lockfile does not block a later process.
            with PipelineInstance(path):pass

    def test_legacy_or_remote_publisher_is_rejected(self):
        node=SimpleNamespace(count_publishers=lambda _:1)
        with self.assertRaisesRegex(RuntimeError,'existing perception publisher'):
            check_existing_publishers(node,lambda *a,**k:None,.01)

    def test_dashboard_subscriber_does_not_block_startup(self):
        node=SimpleNamespace(count_publishers=lambda _:0)
        check_existing_publishers(node,lambda *a,**k:None,.001)
