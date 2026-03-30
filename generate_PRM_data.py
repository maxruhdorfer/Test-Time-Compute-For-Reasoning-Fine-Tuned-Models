import argparse
import json
import random
from vllm import LLM, SamplingParams
import os
from typing import List
from grading.grader import grade_answer

def split_into_steps(output: str, sep: str="\n\n") -> List[str]:
    return output.split(sep)

def truncate_answer(answer: str) -> str:
    """ Truncate answer string right after the first \boxed{} output to cut all the gibberish after it """
    i = 0
    while i < len(answer):
        # find starting point of first boxed content
        if answer.startswith(r'\boxed{', i):
            i += len(r'\boxed{')
            depth = 1

            # loop through the rest of the string and count curly braces
            while i < len(answer) and depth > 0:
                if answer[i] == r"{":
                    depth += 1
                elif answer[i] == r"}":
                    depth -= 1
                i += 1
            break
        else:
            i += 1
    return answer[:i+2] if i+2 < len(answer) else answer[:i]


def extract_boxed(answer: str) -> str:
    """ Extract boxed content in answer while taking into account that due to Latex code there can be additional curly braces 
        Assumes a RAW string as input """
    box_content = r""
    i=0
    while i < len(answer):
        # find starting point
        if answer.startswith(r'\boxed{', i):
            i += len(r'\boxed{')
            depth = 1

            # loop through the rest of the string and count curly braces
            while i < len(answer) and depth > 0:
                if answer[i] == r"{":
                    depth += 1
                elif answer[i] == r"}":
                    depth -= 1
                if depth > 0:
                    box_content += answer[i]
                i += 1
            break
        else:
            i += 1
    return box_content

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="data/PRM_Train/data_selection.json")
    parser.add_argument("--prompt_path", type=str, default="prompts/CoT.prompt")
    parser.add_argument("--model", type=str, default="7-B")
    parser.add_argument("--output", type=str, default="data/PRM_Train/7B/PRM_7B_data.jsonl")
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--rollouts_MC", type=int, default=8)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # load prompt
    with open(args.prompt_path, "r") as file:
        prompt_template = file.read()
    
    # load train data
    # train_data = []
    with open(args.train_dataset, 'r') as json_file:
        train_data = json.load(json_file)
    # with open(args.train_dataset, "r") as file:
    #     for line in file:
    #         train_data.append(json.loads(line))
    
    # sample required number of examples from dataset
    if args.train_samples < len(train_data):
        train_data = random.sample(train_data, args.train_samples)
    
    with open('data/PRM_Train/data_selection.json', 'w') as json_file:
        json.dump(train_data, json_file, indent=4)
        
    # prepare queries
    queries = [prompt_template.format(question=q["problem"]) for q in train_data]
    gt_train = [td["answer"] for td in train_data]

    if args.model == "1-5-B":
        modelCode = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif args.model == "7-B":
        modelCode = "Qwen/Qwen2.5-Math-7B-Instruct"
    else:
        raise ValueError("At the moment only Qwen2.5-Math-1.5B-Instruct and Qwen2.5-Math-7B-Instruct are supported.")
    # initialize model
    llm = LLM(model=modelCode, enable_prefix_caching=True, enable_chunked_prefill=True, max_num_batched_tokens=2048, dtype="bfloat16")

    sampling_params = SamplingParams(
        temperature=args.sampling_temperature, top_p=args.top_p, max_tokens=args.max_tokens, n=args.rollouts
    )

    print("Generating Steps and Answers")

    out = llm.generate(queries, sampling_params)

    results = [
        {
            "prompt": o.prompt,
            "steps": split_into_steps(truncate_answer(completion.text)),
            "answer": extract_boxed(truncate_answer(completion.text)),
            "gt" : gt,
            "correct": grade_answer(extract_boxed(completion.text), gt),
        }
        for o, gt in zip(out, gt_train)
        for completion in o.outputs
    ]

    # Do a second pass and find MC prediction for each step (using MC_rollouts additional generation steps for each step)
    sampling_params_MC = SamplingParams(
        temperature=args.sampling_temperature, top_p=args.top_p, max_tokens=1024, n=args.rollouts_MC
    )

    print("Obtaining MC statistics")
    print("Generating prompts...")
    queries_mc = []
    for prob_idx, res in enumerate(results):
        for step_idx in range(len(res["steps"])-1):
            prefix = (res["prompt"] + "\n\n" +
                      "\n\n".join(res["steps"][:step_idx + 1]) +
                      "\n\n")  # continuation prompt
            queries_mc.append((prob_idx, step_idx, prefix))
        
        # if there is only one step add statistics right now
        if len(res["steps"]) == 1:
            results[prob_idx]["statistics"] = [float(res["correct"])]
    
    prompts_mc = [q[2] for q in queries_mc]

    print("Running MC prompt generation")
    out_mc = llm.generate(prompts_mc, sampling_params_MC)

    # Reconstruct MC labels
    for (prob_idx, step_idx, _), output in zip(queries_mc, out_mc):
        grades = [grade_answer(extract_boxed(c.text), results[prob_idx]["gt"]) for c in output.outputs]
        if "statistics" not in results[prob_idx]:
            results[prob_idx]["statistics"] = []
        
        results[prob_idx]["statistics"].append(sum(grades)/len(grades))
        if step_idx == len(results[prob_idx]["steps"]) - 2:
            results[prob_idx]["statistics"].append(float(results[prob_idx]["correct"]))

    for r in results[:2]:
        print(f"Prompt: {r["prompt"]}")
        for i, step in enumerate(r["steps"]):
            print(f"\n\nStep {i}: {step}")
        print(f"\n\nAnswer: {r["answer"]}\nGround Truth: {r["gt"]}\nCorrect: {r["correct"]}")
        print(f"Statistics: {r["statistics"]}")
    
    print("Saving results")
    file_path = args.output
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    llm.llm_engine.engine_core.shutdown()