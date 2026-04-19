""" Evaluate MATH test set as a Benchmark for several inference techniques and compute constraints """
import argparse
import json
import random
from vllm import LLM, SamplingParams
import os
from typing import List
from grading.grader import grade_answer
from generate_PRM_data import split_into_steps, truncate_answer, extract_boxed
from train_PRM import load_tokenizer
from PRM_model import PRM, score_trace
import torch
from unittest.mock import patch
from inference import majority_vote, vanilla_best_of_N, weighted_best_of_N, beam_search
from tqdm import tqdm
from transformers import AutoModelForCausalLM

STEP_SEPARATOR = '\n<step>\n'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", type=str, default="data/MATH/test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/CoT.prompt")
    parser.add_argument("--model", type=str, default="1-5-B")
    parser.add_argument("--prm_path", type=str, default="checkpoints/PRM_1.5B_Train")
    parser.add_argument("--output_path", type=str, default="logs/benchmark/1.5B/")
    parser.add_argument("--sampling_temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--beam_M", type=int, default=2)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # get device
    if torch.cuda.is_available():
            device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # load prompt
    with open(args.prompt_path, "r") as file:
        prompt_template = file.read()
    
    test_data = []
    with open(args.test_dataset, "r") as file:
        for line in file:
            test_data.append(json.loads(line))
    
    # prepare queries
    queries = [prompt_template.format(question=q["problem"]) for q in test_data]
    gt_test = [td["answer"] for td in test_data]

    if args.model == "1-5-B":
        modelCode = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif args.model == "7-B":
        modelCode = "Qwen/Qwen2.5-Math-7B-Instruct"
    else:
        raise ValueError("At the moment only Qwen2.5-Math-1.5B-Instruct and Qwen2.5-Math-7B-Instruct are supported.")
    
    # initialize model
    llm = LLM(model=modelCode, enable_prefix_caching=True, enable_chunked_prefill=True, max_num_batched_tokens=2048, dtype="bfloat16")

    sampling_params = SamplingParams(
        temperature=args.sampling_temperature, top_p=args.top_p, max_tokens=args.max_tokens, n=args.rollouts,  stop=["<|im_end|>"]
    )

    out = llm.generate(queries, sampling_params)

    results = [
        [{
            "prompt": o.prompt,
            "steps": split_into_steps(truncate_answer(completion.text)),
            "answer": extract_boxed(truncate_answer(completion.text)),
            "gt" : gt,
            "correct": grade_answer(extract_boxed(completion.text), gt),
        } for completion in o.outputs]
        for o, gt in zip(out, gt_test)
    ]

    # free vLLM GPU memory before loading PRM
    del llm
    torch.cuda.empty_cache()

    print(results[0])
    # check majority vote
    correct_maj = 0
    for r in tqdm(results, desc='Evaluate Majority Vote'):
        r_res = majority_vote(r)
        if r_res['correct']:
            correct_maj += 1
        
    print(f"Accuracy for majority vote: {correct_maj/len(results):.3f}")

    # load PRM
    prm_model = PRM.load(args.prm_path, freeze_model = False, device=device)
    tokenizer = load_tokenizer(modelCode)
    print("Successfully loaded the PRM Model")

    correct_vanilla_N = 0
    for r in tqdm(results, desc='Evaluate vanilla best of N'):
        r_res = vanilla_best_of_N(prm_model, tokenizer, STEP_SEPARATOR, r, device)
        if r_res['correct']:
            correct_vanilla_N += 1
        
    print(f"Accuracy for vanilla best of N: {correct_vanilla_N/len(results):.3f}")

    correct_weighted_N = 0
    for r in tqdm(results, desc='Evaluate weighted best of N'):
        r_res = weighted_best_of_N(prm_model, tokenizer, STEP_SEPARATOR, r, device)
        if r_res['correct']:
            correct_weighted_N += 1
        
    print(f"Accuracy for weighted best of N: {correct_weighted_N/len(results):.3f}")

    # beam search – requires step-by-step generation via HuggingFace
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        modelCode,
        torch_dtype=model_dtype,
        device_map=device,
    )

    correct_beam = 0
    for prompt, gt in tqdm(zip(queries, gt_test), total=len(queries), desc="Evaluate Beam Search"):
        r_res = beam_search(
            gen_model=llm,
            prm_model=prm_model,
            tokenizer=tokenizer,
            step_sep=STEP_SEPARATOR,
            prompt=prompt,
            ground_truth=gt,
            N=args.rollouts,
            M=args.beam_M,
            device=device,
        )
        if r_res["correct"]:
            correct_beam += 1

    print(f"Accuracy for beam search (N={args.rollouts}, M={args.beam_M}): {correct_beam/len(queries):.3f}")