""" Evaluate MATH test set as a Benchmark for several inference techniques and compute constraints """
import argparse
import json
from math import e
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

STEP_SEPARATOR = '\n<step>\n'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", type=str, default="data/MATH/test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/CoT.prompt")
    parser.add_argument("--model", type=str, default="1-5-B")
    parser.add_argument("--prm_path_15", type=str, default="checkpoints/PRM_1.5B_Train")
    parser.add_argument("--prm_path_7", type=str, default="checkpoints/PRM_1p5B_7B_Train")
    parser.add_argument("--output_path", type=str, default="logs/benchmark/7B/")
    parser.add_argument("--sampling_temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--beam_M", type=int, default=4)
    
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
    prm_model_15 = PRM.load(args.prm_path_15, freeze_model = False, device=device)
    prm_model_7 = PRM.load(args.prm_path_7, freeze_model = False, device=device)
    tokenizer = load_tokenizer(modelCode)
    print("Successfully loaded the PRM Models")

    correct_vanilla_N_15 = 0
    correct_vanilla_N_7 = 0
    for r in tqdm(results, desc='Evaluate vanilla best of N'):
        r_res15 = vanilla_best_of_N(prm_model_15, tokenizer, STEP_SEPARATOR, r, device)
        r_res7 = vanilla_best_of_N(prm_model_7, tokenizer, STEP_SEPARATOR, r, device)
        if r_res15['correct']:
            correct_vanilla_N_15 += 1
        if r_res7['correct']:
            correct_vanilla_N_7 += 1
        
    print(f"Accuracy for vanilla best of N: 1.5B {correct_vanilla_N_15/len(results):.3f}, 7B {correct_vanilla_N_7/len(results):.3f}")

    correct_weighted_N_15 = 0
    correct_weighted_N_7 = 0
    for r in tqdm(results, desc='Evaluate weighted best of N'):
        r_res15 = weighted_best_of_N(prm_model_15, tokenizer, STEP_SEPARATOR, r, device)
        r_res7 = weighted_best_of_N(prm_model_7, tokenizer, STEP_SEPARATOR, r, device)
        if r_res15['correct']:
            correct_weighted_N_15 += 1
        if r_res7['correct']:
            correct_weighted_N_7 += 1
        
    print(f"Accuracy for weighted best of N: 1.5B {correct_weighted_N_15/len(results):.3f}  7B {correct_weighted_N_7/len(results):.3f}")

    # beam search – re-initialize vLLM with reduced GPU memory to share with PRM
    llm = LLM(model=modelCode, enable_prefix_caching=True, enable_chunked_prefill=True,
              max_num_batched_tokens=2048, dtype="bfloat16", gpu_memory_utilization=0.4)

    if args.beam_M <= args.rollouts:
        counter = 0
        correct_beam15 = 0
        for prompt, gt in tqdm(zip(queries, gt_test), total=len(queries), desc="Evaluate Beam Search"):
            print(f"Rollout {counter}")
            counter += 1
            r_res = beam_search(
                llm=llm,
                prm_model=prm_model_15,
                tokenizer=tokenizer,
                step_sep=STEP_SEPARATOR,
                prompt=prompt,
                ground_truth=gt,
                N=args.rollouts,
                M=args.beam_M,
                device=device,
            )
            if r_res["correct"]:
                correct_beam15 += 1
    else:
        correct_beam15 = 0
    
    if args.beam_M <= args.rollouts:
        correct_beam7 = 0
        counter = 0
        for prompt, gt in tqdm(zip(queries, gt_test), total=len(queries), desc="Evaluate Beam Search"):
            print(f"Rollout {counter}")
            counter += 1
            r_res = beam_search(
                llm=llm,
                prm_model=prm_model_7,
                tokenizer=tokenizer,
                step_sep=STEP_SEPARATOR,
                prompt=prompt,
                ground_truth=gt,
                N=args.rollouts,
                M=args.beam_M,
                device=device,
            )
            if r_res["correct"]:
                correct_beam7 += 1
    else:
        correct_beam7 = 0

    print(f"Accuracy for beam search (N={args.rollouts}, M={args.beam_M}): 1.5B {correct_beam15/len(queries):.3f} 7B {correct_beam7/len(queries):.3f}")

    os.makedirs(args.output_path, exist_ok=True)
    results_data = {
        "rollouts": args.rollouts,
        "beam_M": args.beam_M,
        "accuracies": {
            "majority_vote": correct_maj / len(results),
            "vanilla_best_of_N_15": correct_vanilla_N_15 / len(results),
            "weighted_best_of_N_15": correct_weighted_N_15 / len(results),
            "beam_search_15": correct_beam15 / len(queries),
            "vanilla_best_of_N_7": correct_vanilla_N_7 / len(results),
            "weighted_best_of_N_7": correct_weighted_N_7 / len(results),
            "beam_search_7": correct_beam7 / len(queries),
        },
    }
    output_file = os.path.join(args.output_path, f"results_rollouts{args.rollouts}_beamM{args.beam_M}.json")
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {output_file}")