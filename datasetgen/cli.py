"""Command-line interface for DatasetGen AI."""

from __future__ import annotations

import json
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from datasetgen.config.settings import get_settings
from datasetgen.pipeline.orchestrator import DatasetPipeline
from datasetgen.providers import get_provider
from datasetgen.schemas.blueprint import Blueprint

app = typer.Typer(
    name="datasetgen",
    help="Synthetic dataset generation, programmatic validation, and trace logging factory.",
)
console = Console()


@app.command()
def plan(
    request: str = typer.Argument(..., help="Natural language dataset request"),
    provider: str = typer.Option("mock", "--provider", "-p", help="Provider name (mock, openai, gemini, hf_local)"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Path to fine-tuned LoRA adapter"),
    count: Optional[int] = typer.Option(None, "--count", "-c", help="Override example count"),
) -> None:
    """Analyze a request and produce a structured Blueprint."""
    console.print(f"[bold cyan]Planning dataset request:[/bold cyan] '{request}'")
    provider_kwargs = {"adapter_path": adapter} if adapter else {}
    llm = get_provider(provider, **provider_kwargs)
    pipeline = DatasetPipeline(llm)

    overrides = {}
    if count:
        overrides["number_of_examples"] = count

    blueprint = pipeline.plan(request, overrides=overrides)

    console.print(Panel(
        blueprint.model_dump_json(indent=2),
        title=f"[bold green]Blueprint: {blueprint.dataset_name} (v{blueprint.blueprint_version})[/bold green]",
        border_style="green",
    ))


@app.command()
def run(
    request: str = typer.Argument(..., help="Natural language dataset request"),
    provider: str = typer.Option("mock", "--provider", "-p", help="Provider name (mock, openai, gemini, hf_local)"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Path to fine-tuned LoRA adapter"),
    count: Optional[int] = typer.Option(None, "--count", "-c", help="Override example count"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip blueprint confirmation prompt"),
    batch_size: int = typer.Option(20, "--batch-size", "-b", help="Batch generation size"),
) -> None:
    """Execute end-to-end dataset generation, programmatic validation, and trace logging."""
    settings = get_settings()
    console.print(f"\n[bold blue]=== Starting DatasetGen AI Run ===[/bold blue]")
    console.print(f"[dim]Provider:[/dim] [green]{provider}[/green] | [dim]Data Dir:[/dim] [yellow]{settings.data_dir}[/yellow]\n")

    provider_kwargs = {"adapter_path": adapter} if adapter else {}
    llm = get_provider(provider, **provider_kwargs)
    pipeline = DatasetPipeline(llm, batch_size=batch_size)

    # 1. Plan
    overrides = {}
    if count:
        overrides["number_of_examples"] = count
    blueprint = pipeline.plan(request, overrides=overrides)

    # 2. Display Blueprint Summary
    table = Table(title="Dataset Blueprint Specifications", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim", width=24)
    table.add_column("Value", style="cyan")

    table.add_row("Dataset Name", blueprint.dataset_name)
    table.add_row("Task Type", blueprint.task_type)
    table.add_row("Domain", f"{blueprint.domain} ({blueprint.education_level or 'standard'})")
    table.add_row("Target Count", str(blueprint.number_of_examples))
    table.add_row("Language", blueprint.language)
    table.add_row("Topics", ", ".join(f"{k} ({int(v*100)}%)" for k, v in blueprint.topics.items()))
    table.add_row("Difficulty", ", ".join(f"{k} ({int(v*100)}%)" for k, v in blueprint.difficulty_distribution.items()))
    table.add_row("Validation Rules", ", ".join(blueprint.validation_rules))
    table.add_row("Output Format", blueprint.output_format)
    console.print(table)

    # 3. Approval Step
    if not yes:
        approved = Confirm.ask("\n[bold yellow]Do you approve this Blueprint to start generation?[/bold yellow]")
        if not approved:
            console.print("[red]Generation cancelled by user.[/red]")
            raise typer.Exit(code=0)
    else:
        console.print("\n[bold green][OK] Auto-approved Blueprint via --yes flag[/bold green]")

    # 4. Execute Pipeline
    console.print("\n[bold cyan]Generating batches and applying programmatic validators...[/bold cyan]")
    accepted_examples, report, trace = pipeline.run(
        user_request=request,
        blueprint=blueprint,
    )

    # 5. Output Report
    console.print("\n" + report.format_cli())
    console.print(f"[bold green][OK] Generated {len(accepted_examples)} validated examples saved to:[/bold green]")
    console.print(f"  [cyan]{settings.datasets_dir / blueprint.dataset_name / 'final.jsonl'}[/cyan]")
    console.print(f"[bold green][OK] Full execution trace saved to:[/bold green]")
    console.print(f"  [cyan]{settings.traces_dir / f'trace_{trace.trace_id}.json'}[/cyan]\n")


@app.command()
def bench(
    provider: str = typer.Option("mock", "--provider", "-p", help="Provider name (mock, openai, gemini, hf_local)"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Path to fine-tuned LoRA adapter"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Number of benchmark tasks to run"),
) -> None:
    """Run benchmark evaluation suite across models and categories."""
    from datasetgen.evaluation.benchmark_runner import BenchmarkRunner

    console.print(f"[bold cyan]Running benchmark evaluation suite using provider:[/bold cyan] [green]{provider}[/green]")
    provider_kwargs = {"adapter_path": adapter} if adapter else {}
    llm = get_provider(provider, **provider_kwargs)
    runner = BenchmarkRunner(llm)
    results = runner.run_all(limit=limit)

    table = Table(title="Benchmark Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Benchmark ID", style="cyan")
    table.add_column("Accepted", style="green")
    table.add_column("Acceptance %", style="yellow")
    table.add_column("Schema %", style="green")
    table.add_column("Domain %", style="green")
    table.add_column("Duplicates %", style="red")
    table.add_column("Gate Status", style="bold")

    for r in results:
        status_style = "[bold green]PASSED[/bold green]" if r.passed_quality_gate else "[bold red]FAILED[/bold red]"
        table.add_row(
            r.benchmark_id,
            str(r.total_accepted),
            f"{r.acceptance_rate*100:.1f}%",
            f"{r.schema_valid_rate*100:.1f}%",
            f"{r.domain_valid_rate*100:.1f}%",
            f"{r.duplicate_rate*100:.1f}%",
            status_style,
        )
    console.print(table)
    console.print(f"[bold green][OK] Detailed reports generated in:[/bold green] [cyan]{get_settings().benchmark_results_dir}[/cyan]\n")


@app.command()
def export_sft(
    train_ratio: float = typer.Option(0.90, "--train-ratio", "-r", help="Fraction of records for training split"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for data shuffling"),
) -> None:
    """Export quality-gate-passing traces into SFT training datasets."""
    from datasetgen.training.prepare_sft import SFTDataPreparer

    console.print("[bold cyan]Exporting validated traces into multi-task SFT training dataset...[/bold cyan]")
    preparer = SFTDataPreparer()
    train_p, val_p, summary = preparer.prepare_dataset(train_ratio=train_ratio, seed=seed)

    table = Table(title="SFT Dataset Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric / Task", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Total SFT Records", str(summary["total_records"]))
    table.add_row("Training Split (90%)", str(summary["train_count"]))
    table.add_row("Validation Split (10%)", str(summary["val_count"]))
    table.add_row("Task: Request -> Blueprint", str(summary["tasks"]["request_to_blueprint"]))
    table.add_row("Task: Blueprint -> Batch", str(summary["tasks"]["blueprint_to_batch"]))
    table.add_row("Task: Error Repair", str(summary["tasks"]["error_repair"]))
    table.add_row("Task: Targeted Generation", str(summary["tasks"]["targeted_generation"]))

    console.print(table)
    console.print(f"[bold green][OK] Training set saved to:[/bold green] [cyan]{train_p}[/cyan]")
    console.print(f"[bold green][OK] Validation set saved to:[/bold green] [cyan]{val_p}[/cyan]\n")


@app.command()
def export_dpo(
    train_ratio: float = typer.Option(0.90, "--train-ratio", "-r", help="Fraction of pairs for training split"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for data shuffling"),
) -> None:
    """Export preference pairs (chosen vs rejected) from decision traces for DPO training."""
    from datasetgen.training.prepare_dpo import DPODataPreparer

    console.print("[bold cyan]Exporting validated traces into DPO preference pairs...[/bold cyan]")
    preparer = DPODataPreparer()
    train_p, val_p, summary = preparer.prepare_dataset(train_ratio=train_ratio, seed=seed)

    table = Table(title="DPO Dataset Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Total Preference Pairs", str(summary["total_pairs"]))
    table.add_row("Training Split (90%)", str(summary["train_count"]))
    table.add_row("Validation Split (10%)", str(summary["val_count"]))

    console.print(table)
    console.print(f"[bold green][OK] DPO Training set saved to:[/bold green] [cyan]{train_p}[/cyan]")
    console.print(f"[bold green][OK] DPO Validation set saved to:[/bold green] [cyan]{val_p}[/cyan]\n")


@app.command()
def merge(
    base_model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--base-model", "-m", help="Hugging Face base model ID"),
    adapter: str = typer.Option(..., "--adapter", "-a", help="Path to fine-tuned LoRA adapter"),
    output: str = typer.Option(..., "--output", "-o", help="Target directory for standalone merged model"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for merge operations (cpu or cuda)"),
) -> None:
    """Merge LoRA adapter into base model weights for zero-dependency standalone deployment."""
    from datasetgen.training.merge_adapter import merge_lora_to_standalone

    console.print(f"[bold cyan]Merging LoRA adapter '{adapter}' into '{base_model}'...[/bold cyan]")
    out_dir = merge_lora_to_standalone(
        base_model_id=base_model,
        adapter_path=adapter,
        output_path=output,
        device=device,
    )
    console.print(f"[bold green][OK] Standalone merged model exported to:[/bold green] [cyan]{out_dir}[/cyan]\n")


if __name__ == "__main__":
    app()

