import torch
from PRM_model import PRM
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
import argparse
import json
import wandb
import random
import os

STEP_SEPARATOR = '\n<step>\n'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")    
    parser.add_argument("--train_data_path", type=str, default="data/PRM_Train/7B/PRM_7B_data.jsonl") 
    parser.add_argument("--output", type=str, default="logs/1.5B_model_7B_train.log") 
    parser.add_argument("--run_name", type=str, default="PRM_1p5B_7B_Train") 
    parser.add_argument("--max_tokens", type=int, default=2048)  
    parser.add_argument("--val_fraction", type=float, default=0.1)    
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--val_interval", type=int, default=500)
    # parser.add_argument("--val_steps", type=int, default=40)     
    parser.add_argument("--epochs", type=int, default=3)    
    parser.add_argument("--lr", type=float, default=2e-5)    
    parser.add_argument("--warmup_ratio", type=float, default=0.1)    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--no_checkpoint", action='store_true')
    parser.add_argument("--use_wandb", action='store_true')
    args = parser.parse_args()
    return args

def init_wandb(project: str, run_name: str,config: dict) -> None:
    """Initialize wandb with environment variable support.

    Args:
        project:        wandb project name to use
        run_name:       run name
        config:         Training config to log

    Returns:
        True if wandb is enabled, False otherwise
    """
    wandb.init(
        project=project,
        name=run_name,
        config=config
        )

def log_metrics(metrics: dict, step: int | None = None):
    """Log metrics to wandb."""
    wandb.log(metrics, step=step)


def finish_wandb():
    """Finish wandb run."""
    wandb.finish()

def create_optimizer(
    model: torch.nn.Module,
    lr: float,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer for trainable parameters."""
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with proper padding setup."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def prepare_datapoint(datapoint: dict, tokenizer: AutoTokenizer, step_sep_ids: list[int]) -> dict:
    """ Takes a raw datapoint and assembles it in the format needed for training 
    
        Args:
            datapoint:          dict of form {'prompt': str, 'steps': list[str], 'gt': str, 'answer': str, 'correct': bool, 'statistics': list[float]}
            tokenizer:          AutoTokenizer object
            step_sep_ids:       tokenized step separator
        
        Returns:
            dict of the form {'prompt_ids': list[int], 'labels': list[int], 'attn_mask': list[int]}
            where the original prompt is concatenated with the steps and the step separator. The labels mask every
            token but the step separators, which are supposed to predict the success probability of a step.
            We assign a positive label if at least one of the MC rollouts starting at a specific step reached the correct result    
    """
    # tokenize prompt (keep BOS only here, suppress it for all subsequent chunks)
    prompt_ids = tokenizer(datapoint['prompt']).input_ids

    # assemble input for prm
    labels = [-100] * len(prompt_ids)
    attn_mask = [1] * len(prompt_ids)
    for i, step in enumerate(datapoint['steps']):
        step_ids = tokenizer(step, add_special_tokens=False).input_ids
        prompt_ids += step_ids + step_sep_ids
        labels += [-100] * len(step_ids) + [-100] * len(step_sep_ids)
        labels[-1] = 1 if datapoint["statistics"][i] > 0 else 0
        attn_mask += [1] * len(step_ids) + [1] * len(step_sep_ids)
    
    return {'prompt_ids': prompt_ids, 'labels': labels, 'attn_mask': attn_mask}


def build_dataset(path: str, tokenizer: AutoTokenizer, step_sep_ids: list[int], val_fraction: float=0.1, token_limit: int|None=None) -> dict:
    """ Build the dataset from the raw PRM train data at path, using the tokenizer 
        
        Args:
            path:           relative path to dataset file
            tokenizer:      used tokenizer
            step_sep_ids:   tokenized step separator
            val_fraction:   fraction of data used for validation dataset
            token_limit:    skip training examples which exceed the token limit
        
        Returns:
            Dictionary that contains Dataset classes {'train': Dataset, 'val': Dataset}
            Each Dataset class is generated from a list of dictionaries, where each dictionary
            contains the keys  'prompt_ids', 'label', 'attn_mask'
    """

    # load data
    with open(path, 'r') as file:
        train_data = json.load(file)
    
    # raise error if we didn't actually load any data
    if len(train_data) < 1:
        raise ValueError("No PRM examples loaded.")
    
    # process data
    train_data_processed = []
    for dp in train_data:
        proc_dp = prepare_datapoint(dp, tokenizer, step_sep_ids)
        
        # skip if token limit is exceeded
        if token_limit and len(proc_dp['prompt_ids']) > token_limit:
            continue

        # add to processed data list
        train_data_processed.append(proc_dp)
    
    num_train_dp = int((1-val_fraction)*len(train_data_processed))
    train_dataset = train_data_processed[:num_train_dp]
    if val_fraction == 0.:
        val_dataset = []
    else:
        val_dataset = train_data_processed[num_train_dp: ]
    
    # generate dataset object and return it
    return {'train': Dataset.from_list(train_dataset), 'val': Dataset.from_list(val_dataset)}

def train_prm(
    model: PRM,
    train_DL: DataLoader,
    val_DL: DataLoader|None,
    run_name: str,
    batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    seed: int,
    val_interval: int,
    checkpoint_dir: str = "checkpoints",
    no_checkpoint: bool = False,
    use_wandb: bool = True,
) -> dict:
    """Train a Process Reward Model custom dataset.

    Args:
        model:              previously defined PRM model
        samples: Number of PRM800K samples to use
        batch_size:         Training batch size
        grad_accum_steps:   Gradient accumulation steps
        epochs:             Number of training epochs
        lr:                 Learning rate
        warmup_ratio:       Fraction of total steps for linear LR warmup
        seed:               Random seed
        use_wandb:          Whether to log to wandb

    Returns:
        None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device = model.device

    # Initialize wandb
    if use_wandb:
        init_wandb(
            project="PRM_Training",
            run_name=run_name,
            config={
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "epochs": epochs,
                "lr": lr,
                "warmup_ratio": warmup_ratio,
            },
        )

    # Optimizer and LR scheduler with linear warmup
    optimizer = create_optimizer(model, lr)
    total_optimizer_steps = (len(train_DL) // grad_accum_steps) * epochs
    warmup_steps = int(total_optimizer_steps * warmup_ratio)
    scheduler = (
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        if warmup_steps > 0
        else None
    )

    # Mixed precision for memory efficiency
    autocast_enabled = torch.cuda.is_available()

    # Training loop
    global_step = 0
    best_val_acc = -1.0
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_tokens = 0
        optimizer.zero_grad()

        # Accumulators for logging per optimizer step
        accum_loss = 0.0
        accum_correct = 0
        accum_tokens = 0
        accum_microbatches = 0

        for step_idx, batch in enumerate(train_DL):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, logits = model(**batch)

            (loss / grad_accum_steps).backward()

            # Accumulate metrics over the grad_accum window
            accum_loss += loss.item()
            mask = batch["labels"] != -100
            probs = torch.sigmoid(logits[mask])       # (batch_size, context_length), values in [0, 1]
            preds = (probs > 0.5).long()        # (batch_size, context_length), values 0 or 1
            # preds = logits[mask].argmax(dim=-1)
            correct = (preds == batch["labels"][mask]).sum().item()
            tokens = mask.sum().item()
            accum_correct += correct
            accum_tokens += tokens
            accum_microbatches += 1

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_tokens += tokens

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_DL):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log averaged metrics over the full effective batch
                avg_loss = accum_loss / accum_microbatches
                acc = accum_correct / max(1, accum_tokens)
                print(f"Epoch {epoch} step {global_step} | loss {avg_loss:.4f} | acc {acc:.3f}")
                if use_wandb:
                    log_metrics({"loss": avg_loss, "step_accuracy": acc}, step=global_step)

                # Save statistics
                train_loss_list.append(avg_loss)
                train_acc_list.append(acc)
                # Reset accumulators
                accum_loss = 0.0
                accum_correct = 0
                accum_tokens = 0
                accum_microbatches = 0
            
            # evaluate validation set
            if val_DL and ((step_idx + 1) % val_interval == 0 or (step_idx + 1) == len(train_DL)):
                val_loss = 0.0
                val_tokens, val_correct = 0, 0
                model.eval()
                val_batches = 0
                for val_step, batch in enumerate(val_DL):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.no_grad():
                        loss, logits = model(**batch)

                    val_loss += loss.item()
                    mask = batch["labels"] != -100
                    probs = torch.sigmoid(logits[mask])
                    preds = (probs > 0.5).long()
                    correct = (preds == batch["labels"][mask]).sum().item()
                    tokens = mask.sum().item()
                    val_tokens += tokens
                    val_correct += correct
                    val_batches += 1

                val_avg_loss = val_loss / max(1, val_batches)
                val_acc = val_correct / max(1, val_tokens)
                print(f"Validation Set at {epoch} step {global_step} | loss {val_avg_loss:.4f} | acc {val_acc:.3f}")
                if use_wandb:
                    log_metrics({"val_loss": val_avg_loss, "val_accuracy": val_acc}, step=global_step)
                
                # save statistics
                val_loss_list.append(val_avg_loss)
                val_acc_list.append(val_acc)

                if not no_checkpoint and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_path = os.path.join(checkpoint_dir, run_name)
                    model.save(save_path)
                    print(f"  Saved checkpoint (val_acc={val_acc:.3f}) → {save_path}")

                model.train()



        avg_loss = epoch_loss / len(train_DL)
        accuracy = epoch_correct / max(1, epoch_tokens)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Step Accuracy: {accuracy:.3f}")
        if use_wandb:
            log_metrics({"epoch_loss": avg_loss, "epoch_accuracy": accuracy, "epoch": epoch})

    if use_wandb:
        
        finish_wandb()
    return {'train_loss': train_loss_list, 'train_acc': train_acc_list, 'val_loss': val_loss_list, 'val_acc': val_acc_list}

def collate_fn(batch: list[dict], tokenizer: AutoTokenizer) -> dict:
    """Collate function for DataLoader."""
    max_len = max(len(item["prompt_ids"]) for item in batch)
    inputs = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros_like(inputs)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for idx, item in enumerate(batch):
        length = len(item["prompt_ids"])
        inputs[idx, :length] = torch.tensor(item["prompt_ids"], dtype=torch.long)
        attn[idx, :length] = torch.tensor(item["attn_mask"], dtype=torch.long)
        labels[idx, :length] = torch.tensor(item["labels"], dtype=torch.long)

    return {"input_ids": inputs, "attention_mask": attn, "labels": labels}

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

    print(f"Trainable parameters: {prm.count_trainable_params() / 1e6:.2f}M")

    # load tokenizer and tokenize step separator
    tokenizer = load_tokenizer(args.model_id)
    step_sep_tok = tokenizer(STEP_SEPARATOR, add_special_tokens=False).input_ids

    # load train data
    print(f"Building PRM dataset with {100*args.val_fraction}% in the validation set")
    data = build_dataset(
        args.train_data_path, 
        tokenizer, 
        step_sep_tok, 
        val_fraction=args.val_fraction, 
        token_limit=args.max_tokens)
    
    print(f"Loaded {len(data['train'])} training examples and {len(data['val'])} validation examples")
    train_loader = DataLoader(
        data['train'],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(data['train']) > args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_loader = DataLoader(
        data['val'],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(data['val']) > args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    train_stats = train_prm(
        model=prm,
        train_DL=train_loader,
        val_DL=val_loader,
        run_name=args.run_name,
        batch_size=args.batch_size,
        grad_accum_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        val_interval=args.val_interval,
        checkpoint_dir=args.checkpoint_dir,
        no_checkpoint=args.no_checkpoint,
        use_wandb=args.use_wandb,
    )

    # save train stats
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(train_stats, f, indent=2)

    

if __name__ == "__main__":
    main()