# MODEL GENERTION AND TRAINING
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        bs, seq_len, _ = x.shape
        
       
        q = self.w_q(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
     
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            causal = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
            causal = causal.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
     
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 6, max_seq_len: int = 512, dropout: float = 0.1, d_ff_mult: int = 8):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * d_ff_mult, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
       
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        bs, seq_len = x.shape
        
        
        x = self.dropout(self.token_embedding(x) + self.pos_embedding[:, :seq_len, :])
        
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
     
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
