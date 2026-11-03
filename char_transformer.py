"""The Transformer of the notebook in my own PyTorch classes, to check the GPT-2 model."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Self-attention with several heads, as in the attention layer of GPT-2."""

    def __init__(self, config):
        """Build the layers for the queries, keys and values and the output projection."""
        super().__init__()
        self.heads_num = config.n_head
        # GPT-2 keeps the queries, keys and values of all heads in one linear layer,
        # I do the same, so its weights can be copied one to one
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.attn_pdrop)
        self.residual_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, attention_mask):
        """Return a new vector for every position, from itself and the earlier positions."""
        batch_size, tokens_num, embedding_dim = x.shape
        head_size = embedding_dim // self.heads_num
        Q, K, V = self.qkv(x).split(embedding_dim, dim=-1)
        # one row per head: (batch, heads, tokens, head size)
        Q = Q.view(batch_size, tokens_num, self.heads_num, head_size).transpose(1, 2)
        K = K.view(batch_size, tokens_num, self.heads_num, head_size).transpose(1, 2)
        V = V.view(batch_size, tokens_num, self.heads_num, head_size).transpose(1, 2)

        attention_scores = Q @ K.transpose(-2, -1)
        # a position sees only itself and the earlier positions, and never the padding
        ones = torch.ones(tokens_num, tokens_num, dtype=torch.bool, device=x.device)
        causal_mask = torch.tril(ones)
        padding_mask = attention_mask.bool().view(batch_size, 1, 1, tokens_num)
        allowed = causal_mask & padding_mask
        attention_scores = attention_scores.masked_fill(allowed == 0, -torch.inf)
        attention_scores = attention_scores / (head_size ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.attention_dropout(attention_scores)

        out = attention_scores @ V
        # the heads side by side again: (batch, tokens, embedding)
        out = out.transpose(1, 2).reshape(batch_size, tokens_num, embedding_dim)
        return self.residual_dropout(self.projection(out))


class FeedForward(nn.Module):
    """Two linear layers with GELU in between."""

    def __init__(self, config):
        """Build the layers, from n_embd to n_inner and back."""
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            nn.GELU(),
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        """Return the output of the layers."""
        return self.linear_layers(x)


class Block(nn.Module):
    """One Transformer block, each part with a layer norm in front and a residual connection."""

    def __init__(self, config):
        """Build the attention, the feed-forward part and their two layer norms."""
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, attention_mask):
        """Return the output of the block."""
        residual = x
        x = self.multi_head(self.layer_norm_1(x), attention_mask)
        x = x + residual

        residual = x
        x = self.feed_forward(self.layer_norm_2(x))
        return x + residual


class CharTransformer(nn.Module):
    """Decoder-only Transformer for the next character, with the same layers as GPT-2."""

    def __init__(self, config):
        """Build the embeddings, the blocks and the output layer from the GPT-2 config."""
        super().__init__()
        self.token_embedding_layer = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embedding_layer = nn.Embedding(config.n_positions, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)
        blocks = [Block(config) for _ in range(config.n_layer)]
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.unembedding = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # the output layer uses the weights of the token embedding, as in GPT-2
        self.unembedding.weight = self.token_embedding_layer.weight

    def forward(self, token_ids, attention_mask):
        """Return the logits of the next character for every position."""
        batch_size, tokens_num = token_ids.shape
        x = self.token_embedding_layer(token_ids)
        sequence = torch.arange(tokens_num, device=token_ids.device)
        x = x + self.positional_embedding_layer(sequence)
        x = self.dropout(x)
        # the blocks need the mask, so a loop instead of nn.Sequential
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.layer_norm(x)
        return self.unembedding(x)


def copy_weights(own_model, library_model):
    """Copy the weights of the library model into my own model, layer by layer."""
    weights = library_model.state_dict()
    with torch.no_grad():
        own_model.token_embedding_layer.weight.copy_(weights["transformer.wte.weight"])
        own_model.positional_embedding_layer.weight.copy_(weights["transformer.wpe.weight"])
        own_model.layer_norm.weight.copy_(weights["transformer.ln_f.weight"])
        own_model.layer_norm.bias.copy_(weights["transformer.ln_f.bias"])
        for i, block in enumerate(own_model.blocks):
            prefix = f"transformer.h.{i}."
            linear_layers = {
                "attn.c_attn": block.multi_head.qkv,
                "attn.c_proj": block.multi_head.projection,
                "mlp.c_fc": block.feed_forward.linear_layers[0],
                "mlp.c_proj": block.feed_forward.linear_layers[2],
            }
            for name, layer in linear_layers.items():
                # GPT-2 stores the weights of these layers transposed
                layer.weight.copy_(weights[prefix + name + ".weight"].T)
                layer.bias.copy_(weights[prefix + name + ".bias"])
            layer_norms = {"ln_1": block.layer_norm_1, "ln_2": block.layer_norm_2}
            for name, layer_norm in layer_norms.items():
                layer_norm.weight.copy_(weights[prefix + name + ".weight"])
                layer_norm.bias.copy_(weights[prefix + name + ".bias"])
