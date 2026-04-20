import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMultiViewClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_input_x: bool = False,
        fusion: str = "attention",
    ):
        super().__init__()
        if fusion not in {"mean", "attention"}:
            raise ValueError(f"Unsupported fusion mode: {fusion}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.use_input_x = bool(use_input_x)
        self.fusion = fusion

        actual_input_dim = self.input_dim if self.use_input_x else 1
        self.input_proj = nn.Linear(actual_input_dim, self.hidden_dim)
        self.self_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.msg_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(self.num_layers)]
        )
        self.view_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def _prepare_inputs(self, x: torch.Tensor, e: torch.Tensor, node_mask: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if e.dim() == 4:
            e = e.unsqueeze(1)
        if node_mask.dim() == 1:
            node_mask = node_mask.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, N, D] or [N, D], got {x.shape}")
        if e.dim() != 5:
            raise ValueError(f"e must have shape [B, K, N, N, De] or [B, N, N, De], got {e.shape}")
        if node_mask.dim() != 2:
            raise ValueError(f"node_mask must have shape [B, N] or [N], got {node_mask.shape}")
        if x.size(0) != e.size(0) or x.size(0) != node_mask.size(0):
            raise ValueError("Batch dimensions of x, e and node_mask must match")
        if x.size(1) != e.size(2) or x.size(1) != e.size(3) or x.size(1) != node_mask.size(1):
            raise ValueError("Node dimensions of x, e and node_mask must match")
        return x.float(), e.float(), node_mask.bool()

    def _edge_prob(self, e: torch.Tensor) -> torch.Tensor:
        if e.size(-1) == 1:
            a = e.squeeze(-1)
        elif e.size(-1) >= 2:
            a = e[..., 1]
        else:
            raise ValueError(f"Unexpected edge feature dim: {e.size(-1)}")
        a = 0.5 * (a + a.transpose(-1, -2))
        diag_mask = torch.eye(a.size(-1), device=a.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        a = a.masked_fill(diag_mask, 0.0)
        return a

    def forward(self, x: torch.Tensor, e: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        x, e, node_mask = self._prepare_inputs(x, e, node_mask)
        bs, num_nodes = node_mask.shape
        if self.use_input_x:
            if x.size(-1) != self.input_dim:
                raise ValueError(f"Expected x dim {self.input_dim}, got {x.size(-1)}")
            h0 = x
        else:
            h0 = torch.ones(bs, num_nodes, 1, device=x.device, dtype=x.dtype)

        h0 = self.input_proj(h0)
        h0 = h0 * node_mask.unsqueeze(-1)
        a = self._edge_prob(e)

        view_outputs = []
        for view_idx in range(a.size(1)):
            h = h0
            a_view = a[:, view_idx]
            a_view = a_view * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            deg = a_view.sum(dim=-1, keepdim=True).clamp(min=1.0)

            for self_layer, msg_layer in zip(self.self_layers, self.msg_layers):
                msg = torch.bmm(a_view, h) / deg
                h = self_layer(h) + msg_layer(msg)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = h * node_mask.unsqueeze(-1)
            view_outputs.append(h)

        view_stack = torch.stack(view_outputs, dim=1)
        if self.fusion == "mean" or view_stack.size(1) == 1:
            h = view_stack.mean(dim=1)
        else:
            attn_logits = self.view_gate(view_stack).squeeze(-1)
            attn = torch.softmax(attn_logits, dim=1)
            h = (attn.unsqueeze(-1) * view_stack).sum(dim=1)

        h = h * node_mask.unsqueeze(-1)
        logits = self.classifier(h)
        return logits

    def save_checkpoint(self, path: str):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_classes": self.num_classes,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "use_input_x": self.use_input_x,
                    "fusion": self.fusion,
                },
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, map_location: Optional[str] = "cpu"):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
