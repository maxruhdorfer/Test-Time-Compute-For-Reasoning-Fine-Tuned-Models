""" Define PRM Model """
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

from typing import Callable, Iterator, Any, Dict, List

class PRM(nn.Module):
    """ Class for the PRM. It uses a small pre-trained LM, e.g. QWEN2.5 with 1.5B parameters and a classification head

    The model weights can be frozen such that either only the classification head can be trained independently  
    or the whole model using PEFT methods such as LORA. """

    def __init__(
        self,
        model_id: str,
        head_dim: int=1,
        freeze_model: bool=True,
        device: torch.device | str = "cpu",
        lora_k: int=16,
        lora_alpha: int=32,
        lora_dropout: float=0.1
    ) -> None:

        super().__init__()

        # bfloat16 backward is not supported on MPS
        device_str = device if isinstance(device, str) else device.type
        model_dtype = torch.bfloat16 if device_str == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            device_map=device,
        )

        # Turn of KV caching for training
        self.model.config.use_cache = False

        # optionally freeze model parameters
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        # Build head with same dtype as model
        self.head = nn.Linear(self.model.config.hidden_size, head_dim, bias=False)
        self.head = self.head.to(model_dtype)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """ perform a forward pass through the network and return the output logits, as well as the cross-entropy loss (or None) 
        
        Args:
            input_ids:          input tokens [batch_size, seq_len]
            attention_mask:     attention mask [batch_size, seq_len]
            labels:             labels for prediction, 0 and 1 for steps and -100 for masked tokens
        
        Returns:
            loss:               cross-entropy loss on step tokens or None if no labels are given
            logits:             per token class logits [batch_size, seq_len, 1]        
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]

        logits = self.head(hidden)
        logits_squeezed = logits.squeeze(-1)

        loss = None
        if labels is not None:
            mask = labels != -100
            loss = F.binary_cross_entropy_with_logits(logits_squeezed[mask], labels[mask].float())

        return loss, logits_squeezed

    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)