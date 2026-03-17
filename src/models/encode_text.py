import torch
import torch.nn as nn
from transformers import AutoModel


class SharedEncoder(nn.Module):

    def __init__(self, model_name="sbintuitions/modernbert-ja-310m"):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        # enable gradient checkpointing
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS pooling
        cls_embedding = outputs.last_hidden_state[:, 0]

        return cls_embedding