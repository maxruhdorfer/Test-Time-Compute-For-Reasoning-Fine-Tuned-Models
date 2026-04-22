"""
Beam-search debugger: runs one problem through beam_search and prints a
step-by-step tree of every candidate, its PRM score, and which beams survived.

Usage:
    python visualize_beam_search.py \
        --question "What is 2+2?" \
        --ground_truth "4" \
        --model 1-5-B \
        --prm_path checkpoints/PRM_1.5B_Train \
        --rollouts 8 \
        --beam_M 4

Or pipe a question from the MATH test set:
    python visualize_beam_search.py --problem_idx 0
"""

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

STEP_SEPARATOR = "\n<step>\n"

# ─── debug-instrumented beam search ──────────────────────────────────────────

@dataclass
class StepSnapshot:
    depth: int
    candidates: List[Dict]          # all expanded candidates (N total)
    survivors: List[Dict]           # top beam_width that survived
    pruned: List[Dict]              # the ones that got dropped


def beam_search_debug(
    llm,
    prm_model,
    tokenizer,
    step_sep: str,
    prompt: str,
    ground_truth: str,
    N: int,
    M: int,
    max_steps: int = 40,
    device="cpu",
    temperature: float = 0.7,
    gen_step_sep: str = "\n\n",
    max_model_len: int = 4096,
    max_new_tokens: int = 512,
) -> tuple[Dict, List[StepSnapshot]]:
    """Identical to inference.beam_search but also returns per-depth snapshots."""
    from inference import _generate_next_steps_vllm, _score_candidates_batched
    from generate_PRM_data import extract_boxed
    from grading.math_normalize import normalize_answer
    from grading.grader import grade_answer

    beam_width = N // M
    beams: List[Dict] = [{"steps": [], "score": 0.0, "done": False} for _ in range(beam_width)]
    token_budget = max_model_len - max_new_tokens
    snapshots: List[StepSnapshot] = []

    for depth in range(max_steps):
        active = [b for b in beams if not b["done"]]
        done_beams = [b for b in beams if b["done"]]
        if not active:
            break

        for beam in active:
            ctx = prompt + "".join(s + gen_step_sep for s in beam["steps"])
            if len(tokenizer.encode(ctx)) >= token_budget:
                beam["done"] = True

        active = [b for b in active if not b["done"]]
        done_beams = [b for b in beams if b["done"]]
        if not active:
            break

        contexts = [
            prompt + "".join(s + gen_step_sep for s in beam["steps"])
            for beam in active
            for _ in range(M)
        ]
        all_steps = _generate_next_steps_vllm(
            llm, contexts, gen_step_sep, max_new_tokens=max_new_tokens, temperature=temperature
        )

        new_candidates: List[Dict] = []
        for i, beam in enumerate(active):
            for j in range(M):
                step = all_steps[i * M + j]
                new_steps = beam["steps"] + [step]
                is_done = r"\boxed{" in step or r"\boxed {" in step
                new_candidates.append({
                    "steps": new_steps,
                    "score": 0.0,
                    "done": is_done,
                    "parent_beam": i,
                    "proposal_idx": j,
                })

        scores = _score_candidates_batched(prm_model, tokenizer, prompt, new_candidates, step_sep, device)
        for cand, score in zip(new_candidates, scores):
            cand["score"] = score

        all_candidates = done_beams + new_candidates
        all_candidates.sort(key=lambda c: c["score"], reverse=True)
        survivors = all_candidates[:beam_width]
        survivor_ids = {id(s) for s in survivors}
        pruned = [c for c in all_candidates[beam_width:] if c not in done_beams]

        snapshots.append(StepSnapshot(
            depth=depth,
            candidates=deepcopy(all_candidates),
            survivors=deepcopy(survivors),
            pruned=deepcopy(pruned),
        ))

        beams = survivors

        if all(b["done"] for b in beams):
            break

    best = max(beams, key=lambda b: b["score"])
    last_step = best["steps"][-1] if best["steps"] else ""
    answer_text = last_step if (r"\boxed{" in last_step or r"\boxed {" in last_step) else "".join(best["steps"])
    answer = normalize_answer(extract_boxed(answer_text))
    correct = grade_answer(answer, ground_truth)

    result = {
        "beam_answer": answer,
        "beam_score": best["score"],
        "beam_steps": len(best["steps"]),
        "correct": correct,
        "best_trace": best["steps"],
    }
    return result, snapshots


# ─── rendering ───────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 90) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.splitlines()) if text else ""


def _step_label(cand: Dict) -> str:
    parent = cand.get("parent_beam", "?")
    prop = cand.get("proposal_idx", "?")
    return f"beam{parent}·prop{prop}"


def render_snapshot(snap: StepSnapshot, beam_width: int) -> None:
    console.rule(f"[bold cyan]Depth {snap.depth}  —  {len(snap.candidates)} candidates → keep {beam_width}[/bold cyan]")

    survivor_ids = {id(c) for c in snap.survivors}

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        expand=True,
        min_width=100,
    )
    table.add_column("Rank", style="dim", width=5, justify="right")
    table.add_column("Label", width=14)
    table.add_column("PRM score", width=10, justify="right")
    table.add_column("Done?", width=6, justify="center")
    table.add_column("Last step (truncated)", no_wrap=False)

    for rank, cand in enumerate(snap.candidates, 1):
        survived = id(cand) in survivor_ids or any(
            c["steps"] == cand["steps"] for c in snap.survivors
        )
        score = cand["score"]
        last_step = cand["steps"][-1] if cand["steps"] else ""
        truncated = _wrap(last_step[:400] + ("…" if len(last_step) > 400 else ""), 80)
        done_mark = "[green]✓[/green]" if cand.get("done") else ""
        label = _step_label(cand)

        if survived and rank <= beam_width:
            row_style = "green"
            rank_str = f"[bold green]★{rank}[/bold green]"
        else:
            row_style = "dim red"
            rank_str = f"[dim]{rank}[/dim]"

        table.add_row(
            rank_str,
            f"[{row_style}]{label}[/{row_style}]",
            f"[bold {'green' if survived and rank <= beam_width else 'red'}]{score:.4f}[/bold {'green' if survived and rank <= beam_width else 'red'}]",
            done_mark,
            f"[{row_style}]{truncated}[/{row_style}]",
        )

    console.print(table)


def render_final(result: Dict, question: str) -> None:
    console.rule("[bold yellow]Final Result[/bold yellow]")
    color = "green" if result["correct"] else "red"
    status = "CORRECT ✓" if result["correct"] else "WRONG ✗"

    summary = Table.grid(padding=(0, 2))
    summary.add_row("[bold]Question:[/bold]", _wrap(question, 80))
    summary.add_row("[bold]Answer:[/bold]", result["beam_answer"])
    summary.add_row("[bold]Score:[/bold]", f"{result['beam_score']:.4f}")
    summary.add_row("[bold]Steps:[/bold]", str(result["beam_steps"]))
    summary.add_row("[bold]Status:[/bold]", f"[bold {color}]{status}[/bold {color}]")
    console.print(Panel(summary, title="Beam Search Summary", border_style=color))

    console.rule("[dim]Best trace[/dim]")
    for i, step in enumerate(result["best_trace"], 1):
        console.print(Panel(
            _wrap(step, 88),
            title=f"[dim]Step {i}[/dim]",
            border_style="blue",
            padding=(0, 1),
        ))


def visualize(result: Dict, snapshots: List[StepSnapshot], question: str, N: int, M: int) -> None:
    beam_width = N // M
    console.print()
    console.rule("[bold magenta]BEAM SEARCH DEBUG VISUALIZATION[/bold magenta]")
    console.print(f"[dim]N={N}  M={M}  beam_width={beam_width}  depths={len(snapshots)}[/dim]\n")
    console.print(Panel(_wrap(question, 88), title="[bold]Question[/bold]", border_style="magenta"))

    for snap in snapshots:
        render_snapshot(snap, beam_width)

    render_final(result, question)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--question", type=str, default=None, help="Problem text (overrides --problem_idx)")
    p.add_argument("--ground_truth", type=str, default=None)
    p.add_argument("--problem_idx", type=int, default=0, help="Index into data/MATH/test.jsonl")
    p.add_argument("--test_dataset", type=str, default="data/MATH/test.jsonl")
    p.add_argument("--prompt_path", type=str, default="prompts/CoT.prompt")
    p.add_argument("--model", type=str, default="1-5-B", choices=["1-5-B", "7-B"])
    p.add_argument("--prm_path", type=str, default="checkpoints/PRM_1.5B_Train")
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--beam_M", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    # ── resolve question / ground_truth ──────────────────────────────────────
    if args.question:
        question_text = args.question
        ground_truth = args.ground_truth or ""
    else:
        with open(args.test_dataset) as f:
            problems = [json.loads(line) for line in f]
        entry = problems[args.problem_idx]
        question_text = entry["problem"]
        ground_truth = entry["answer"]
        console.print(f"[dim]Loaded problem #{args.problem_idx} from {args.test_dataset}[/dim]")

    with open(args.prompt_path) as f:
        prompt_template = f.read()
    prompt = prompt_template.format(question=question_text)

    # ── load models ──────────────────────────────────────────────────────────
    from vllm import LLM
    from PRM_model import PRM
    from train_PRM import load_tokenizer

    model_id = {
        "1-5-B": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "7-B":   "Qwen/Qwen2.5-Math-7B-Instruct",
    }[args.model]

    console.print(f"[dim]Loading LLM: {model_id}[/dim]")
    llm = LLM(
        model=model_id,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Loading PRM: {args.prm_path}[/dim]")
    prm_model = PRM.load(args.prm_path, freeze_model=False, device=device)
    tokenizer = load_tokenizer(model_id)

    # ── run ──────────────────────────────────────────────────────────────────
    result, snapshots = beam_search_debug(
        llm=llm,
        prm_model=prm_model,
        tokenizer=tokenizer,
        step_sep=STEP_SEPARATOR,
        prompt=prompt,
        ground_truth=ground_truth,
        N=args.rollouts,
        M=args.beam_M,
        max_steps=args.max_steps,
        device=device,
        temperature=args.temperature,
    )

    visualize(result, snapshots, question_text, N=args.rollouts, M=args.beam_M)
