import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

import math


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, block_size,
                       hidden_dim,
                       num_heads,
                       attn_pdrop,
                       resid_pdrop):
        super().__init__()
        assert hidden_dim % num_heads == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                .view(1, 1, block_size, block_size)
        )
        self.n_head = num_heads

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(
            B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(
            B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(
            B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, block_size,
                       hidden_dim,
                       num_heads,
                       attn_pdrop,
                       resid_pdrop):

        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(
            block_size=block_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, block_size,
                       hidden_dim,
                       num_layers,
                       num_heads,
                       attn_pdrop,
                       resid_pdrop,
                       embd_pdrop):

        super().__init__()

        # input embedding stem
        self.pos_emb = nn.Parameter(
            torch.zeros(1, block_size, hidden_dim)
        )
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(
                block_size=block_size,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop
            ) for _ in range(num_layers)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        _, t, _ = x.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(x + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        out = self.head(x)

        return out



class GCN(nn.Module):

    def __init__(self, input_dim,
                       hidden_dim,
                       output_dim,
                       num_layers,
                       dropout_p):

        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dropout_p = dropout_p

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.convs = torch.nn.ModuleList(
            [GCNConv(
                in_channels=dims[i],
                out_channels=dims[i+1]
            ) for i in range(num_layers)]
        )
        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(
                num_features=dims[i+1]
            ) for i in range(num_layers-1)]
        )

    def forward(self, x, edge_index):
        for i in range(self.num_layers-1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.gelu(x)
            # x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x



