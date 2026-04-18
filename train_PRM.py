import torch
from PRM_model import PRM
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer
import argparse
import json
import torch.nn.functional as F

STEP_SEPARATOR = '/n<step>/n'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")    
    parser.add_argument("--train_data_path", type=str, default="data/PRM_Train/1.5B/PRM_1p5B_data_chat.jsonl")    
    args = parser.parse_args()
    return args

def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with proper padding setup."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def main():
    # parse arguments
    args = get_args()

    # get device
    if torch.cuda.is_available():
            device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    # load the model
    prm = PRM(model_id=args.model_id, head_dim=1, device=device, freeze_model=False)
    prm.to(device)

    print(f"Model has {prm.count_trainable_params()} trainable parameters")

    # load the data
    with open(args.train_data_path, 'r') as json_file:
        train_data = json.load(json_file)
    
    print(f"Loaded {len(train_data)} training examples")
    print(train_data[0])

    tokenizer = load_tokenizer(args.model_id)
    step_sep_tok = tokenizer(STEP_SEPARATOR).input_ids
    prompt_ids = tokenizer(train_data[0]['prompt']).input_ids

    # assemble input for prm
    prm_input_ids = prompt_ids
    labels = [-100] * len(prm_input_ids)
    label_ids =[]
    label_vals = []
    for i, step in enumerate(train_data[0]['steps']):
        step_ids = tokenizer(step).input_ids
        prm_input_ids += step_ids + step_sep_tok
        labels += [-100] * len(step_ids) + [-100] * len(step_sep_tok)
        labels[-1] = 1 if train_data[0]["statistics"][i] > 0 else 0
        label_ids.append(len(labels)-1)
        label_vals.append(labels[-1])
    
    labels_tens = torch.tensor(labels).unsqueeze(0).to(device)
    prm_input_ids_tens = torch.tensor(prm_input_ids).unsqueeze(0).to(device)
    att_mask = torch.ones_like(prm_input_ids_tens).to(device)


    print(f"Tokenized prompt has {len(prompt_ids)} tokens")
    print(prm_input_ids)
    print(labels)
    print(tokenizer.decode(prm_input_ids))

    prm.eval()
    with torch.no_grad():
        loss, logits = prm(
            input_ids=prm_input_ids_tens,
            attention_mask=att_mask,
            labels=labels_tens
        )
    print(f"Logits shape: {logits.shape}")
    print(logits)
    print(loss)
    loss_logits = logits[0, label_ids, 0]
    print(loss_logits.shape)
    print("Logits for loss calculation:", loss_logits)
    label_val_tens = torch.tensor(label_vals, dtype=torch.float).to(device)
    print("Labels for loss calculation:", label_val_tens)
    print(f"Loss from model Evaluation: {loss.item():.4f}")
    print(f"Manual loss calculation: {F.binary_cross_entropy_with_logits(loss_logits, label_val_tens, reduction='sum')}")


if __name__ == "__main__":
    main()