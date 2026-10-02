"""FastAPI backend server for DatasetGen AI Studio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from datasetgen.config.settings import get_settings
from datasetgen.pipeline.orchestrator import DatasetPipeline
from datasetgen.providers import get_provider
from datasetgen.schemas.blueprint import Blueprint
from datasetgen.storage.jsonl_io import read_jsonl
from datasetgen.storage.trace_store import TraceStore

app = FastAPI(
    title="DatasetGen AI Studio",
    description="Neuro-symbolic synthetic dataset generation factory with programmatic validators.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


class PlanRequest(BaseModel):
    request: str = Field(..., description="Natural language dataset specification")
    provider: str = Field(default="mock", description="LLM provider name")
    adapter_path: Optional[str] = Field(default=None, description="Optional LoRA adapter checkpoint path")
    overrides: Dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    user_request: str
    blueprint: Blueprint
    provider: str = Field(default="mock")
    adapter_path: Optional[str] = Field(default=None)
    batch_size: int = Field(default=10)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "DatasetGen AI Studio"}


@app.post("/api/plan")
def create_plan(body: PlanRequest) -> Blueprint:
    try:
        kwargs = {"adapter_path": body.adapter_path} if body.adapter_path else {}
        llm = get_provider(body.provider, **kwargs)
        pipeline = DatasetPipeline(llm)
        blueprint = pipeline.plan(body.request, overrides=body.overrides)
        return blueprint
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
def run_generation(body: GenerateRequest) -> Dict[str, Any]:
    try:
        kwargs = {"adapter_path": body.adapter_path} if body.adapter_path else {}
        llm = get_provider(body.provider, **kwargs)
        pipeline = DatasetPipeline(llm, batch_size=body.batch_size)
        accepted, report, trace = pipeline.run(
            user_request=body.user_request,
            blueprint=body.blueprint,
            approval_callback=lambda bp: True,
        )
        return {
            "dataset_name": body.blueprint.dataset_name,
            "accepted_count": len(accepted),
            "report": report.to_dict(),
            "trace_id": trace.trace_id,
            "sample_examples": [ex.data for ex in accepted[:5]],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets")
def list_datasets() -> List[Dict[str, Any]]:
    settings = get_settings()
    datasets_dir = settings.datasets_dir
    results = []
    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.is_dir():
                final_file = d / "final.jsonl"
                count = len(read_jsonl(final_file)) if final_file.exists() else 0
                results.append({
                    "name": d.name,
                    "example_count": count,
                    "has_final": final_file.exists(),
                    "path": str(final_file) if final_file.exists() else None,
                })
    return results


@app.get("/api/datasets/{dataset_name}/download")
def download_dataset(dataset_name: str):
    settings = get_settings()
    file_path = settings.datasets_dir / dataset_name / "final.jsonl"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset final file not found")
    return FileResponse(file_path, filename=f"{dataset_name}.jsonl", media_type="application/jsonlines")


@app.get("/api/traces")
def list_traces(limit: int = Query(20)) -> List[Dict[str, Any]]:
    settings = get_settings()
    store = TraceStore(traces_dir=settings.traces_dir)
    traces = store.list_traces()
    results = []
    for t in traces[:limit]:
        results.append({
            "trace_id": t.trace_id,
            "user_request": t.user_request,
            "dataset_name": t.blueprint.dataset_name,
            "domain": t.blueprint.domain,
            "accepted_count": len(t.final_examples),
            "rejected_count": len(t.rejected_examples),
            "timestamp": t.timestamp.isoformat() if hasattr(t.timestamp, "isoformat") else str(t.timestamp),
        })
    return results


@app.get("/api/traces/{trace_id}")
def get_trace_detail(trace_id: str) -> Dict[str, Any]:
    settings = get_settings()
    store = TraceStore(traces_dir=settings.traces_dir)
    trace = store.load_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump()


# Mount frontend static directory if exists
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
