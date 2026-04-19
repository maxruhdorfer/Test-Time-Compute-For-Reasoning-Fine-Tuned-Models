""" Implement different inference methods """
from PRM_model import PRM, score_trace
import torch
from typing import List, Dict
from collections import Counter, defaultdict
from grading.math_normalize import normalize_answer
from grading.grader import grade_answer
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_PRM_data import extract_boxed

def _generate_next_steps_batched(
    gen_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    contexts: List[str],
    step_sep: str,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[str]:
    """Generate one reasoning step for each context in a single batched call."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(contexts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for out in output_ids:
        new_text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        if step_sep in new_text:
            new_text = new_text[: new_text.index(step_sep)]
        results.append(new_text.strip())
    return results


def _score_candidates_batched(
    prm_model: PRM,
    tokenizer: AutoTokenizer,
    prompt: str,
    candidates: List[Dict],
    step_sep: str,
    device: torch.device,
) -> List[float]:
    """Score all candidates' last step in a single batched PRM forward pass."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    all_input_ids: List[List[int]] = []
    last_step_indices: List[int] = []

    for cand in candidates:
        ids = list(prompt_ids)
        last_idx = 0
        for step_text in cand["steps"]:
            encoded = tokenizer(step_text + step_sep, add_special_tokens=False)["input_ids"]
            ids.extend(encoded)
            last_idx = len(ids) - 1
        all_input_ids.append(ids)
        last_step_indices.append(last_idx)

    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id or 0
    padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in all_input_ids]
    attn_masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_input_ids]

    batch = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long).to(device),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long).to(device),
    }

    prm_model.eval()
    with torch.no_grad():
        _, logits = prm_model(**batch)
        probs = torch.sigmoid(logits)

    return [probs[i, idx].cpu().item() for i, idx in enumerate(last_step_indices)]


def beam_search(
    gen_model: AutoModelForCausalLM,
    prm_model: PRM,
    tokenizer: AutoTokenizer,
    step_sep: str,
    prompt: str,
    ground_truth: str,
    N: int,
    M: int,
    max_steps: int = 40,
    device: torch.device = "cpu",
    temperature: float = 0.7,
    gen_step_sep: str = "\n\n",
) -> Dict[str, str | int | bool]:
    """PRM-guided beam search (arxiv 2408.03314, §3.2).

    At each depth N/M active beams each generate M step proposals → N candidates.
    The PRM scores each candidate by its last-step probability; the top N/M survive.

    Args:
        N:              total rollout budget per step (= beam_width × M)
        M:              step proposals sampled per active beam
        max_steps:      maximum reasoning steps before terminating
        gen_step_sep:   separator the LM naturally produces between steps (used for
                        context assembly and stop-on-separator); distinct from step_sep
                        which is only used by the PRM
    """
    beam_width = N // M
    beams: List[Dict] = [{"steps": [], "score": 0.0, "done": False} for _ in range(beam_width)]

    for _ in range(max_steps):
        active = [b for b in beams if not b["done"]]
        done_beams = [b for b in beams if b["done"]]
        if not active:
            break

        contexts = [
            prompt + "".join(s + gen_step_sep for s in beam["steps"])
            for beam in active
            for _ in range(M)
        ]
        all_steps = _generate_next_steps_batched(
            gen_model, tokenizer, contexts, gen_step_sep, device, temperature=temperature
        )

        new_candidates: List[Dict] = []
        for i, beam in enumerate(active):
            for j in range(M):
                step = all_steps[i * M + j]
                new_steps = beam["steps"] + [step]
                is_done = r"\boxed{" in step or r"\boxed {" in step
                new_candidates.append({"steps": new_steps, "score": 0.0, "done": is_done})

        scores = _score_candidates_batched(prm_model, tokenizer, prompt, new_candidates, step_sep, device)
        for cand, score in zip(new_candidates, scores):
            cand["score"] = score

        all_candidates = done_beams + new_candidates
        all_candidates.sort(key=lambda c: c["score"], reverse=True)
        beams = all_candidates[:beam_width]

    best = max(beams, key=lambda b: b["score"])
    last_step = best["steps"][-1] if best["steps"] else ""
    answer_text = last_step if (r"\boxed{" in last_step or r"\boxed {" in last_step) else "".join(best["steps"])
    answer = normalize_answer(extract_boxed(answer_text))
    correct = grade_answer(answer, ground_truth)

    return {
        "beam_answer": answer,
        "beam_score": best["score"],
        "beam_steps": len(best["steps"]),
        "correct": correct,
    }


def majority_vote(rollout: List[Dict[str, str]]) -> Dict[str, str|int|bool]:
    counts = Counter()
    ground_truth = rollout[0]['gt']

    # loop through rollouts
    for r in rollout:
        norm_answer = normalize_answer(r["answer"])
        counts[norm_answer] += 1
    
    # extract the most common answer and check for correctness
    majority_answer = max(counts, key=counts.get)
    correct = grade_answer(majority_answer, ground_truth)

    return {'maj_answer': majority_answer, 'max_count': counts[majority_answer], 'correct': correct}

def vanilla_best_of_N(prm_model: PRM, tokenizer: AutoTokenizer, step_sep: str, rollout: List[Dict[str, str]], device: torch.device) -> Dict[str, str|int|bool]:
    """ Take score of last step in reasoning trace as score """
    scores = {}
    ground_truth = rollout[0]['gt']

    # loop through rollouts
    for r in rollout:
        norm_answer = normalize_answer(r["answer"])
        prm_res = score_trace(prm_model, tokenizer, r["prompt"], r["steps"], step_sep, device)

        scores[prm_res[-1]["prob"]] = norm_answer

    # find highest score and return result
    max_score = max(scores.keys())
    correct = grade_answer(scores[max_score], ground_truth)
    return {'max_answer': scores[max_score], 'max_score': max_score, 'correct': correct}

def weighted_best_of_N(prm_model: PRM, tokenizer: AutoTokenizer, step_sep: str, rollout: List[Dict[str, str]], device: torch.device) -> Dict[str, str|int|bool]:
    """ Take score of last step in reasoning trace as score """
    scores = defaultdict(float)
    ground_truth = rollout[0]['gt']

    # loop through rollouts
    for r in rollout:
        norm_answer = normalize_answer(r["answer"])
        prm_res = score_trace(prm_model, tokenizer, r["prompt"], r["steps"], step_sep, device)

        scores[norm_answer] += prm_res[-1]["prob"]

    # find highest score and return result
    max_score_answer = max(scores, key=scores.get)
    max_score = scores[max_score_answer]
    correct = grade_answer(max_score_answer, ground_truth)
    return {'max_answer': max_score_answer, 'max_score': max_score, 'correct': correct}
