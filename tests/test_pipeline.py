"""End-to-end integration tests for the dataset generation pipeline."""

import tempfile
from pathlib import Path
from datasetgen.config.settings import Settings
from datasetgen.pipeline.orchestrator import DatasetPipeline
from datasetgen.providers.mock import MockProvider
from datasetgen.storage.checkpoint import CheckpointManager
from datasetgen.storage.trace_store import TraceStore


def test_pipeline_end_to_end_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        provider = MockProvider(seed=123, error_rate=0.20)
        trace_store = TraceStore(traces_dir=tmp_path / "traces")
        chk_mgr = CheckpointManager(checkpoints_dir=tmp_path / "checkpoints")

        pipeline = DatasetPipeline(
            provider=provider,
            checkpoint_manager=chk_mgr,
            trace_store=trace_store,
            batch_size=10,
        )

        request = "I need 20 high-school physics questions"
        blueprint = pipeline.plan(request, overrides={"number_of_examples": 20})

        accepted, report, trace = pipeline.run(
            user_request=request,
            blueprint=blueprint,
            approval_callback=lambda bp: True,
        )

        # Assertions
        assert len(accepted) == 20
        assert report.accepted == 20
        assert report.generated >= 20
        assert report.requested == 20

        # Verify trace contains telemetry
        assert trace.blueprint.dataset_name == blueprint.dataset_name
        assert len(trace.final_examples) == 20
        assert len(trace.validation_results) > 0
        if report.rejected > 0:
            assert len(trace.rejected_examples) == report.rejected
