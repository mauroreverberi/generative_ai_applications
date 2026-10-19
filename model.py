"""Decoder-only Transformer for the company names, built from my own PyTorch classes."""

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    """One attention head that only looks at the current and the earlier characters."""

    def __init__(self, config):
        """Create the query, key and value layers and the mask for the earlier characters."""
        super().__init__()
        self.Q_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])
        self.K_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])
        self.V_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        context_size = config["context_size"]
        causal_attention_mask = torch.tril(torch.ones(context_size, context_size))
        self.register_buffer("causal_attention_mask", causal_attention_mask)

    def forward(self, input):
        """Return the attention output, shape (batch, tokens, head_size)."""
        batch_size, tokens_num, embedding_dim = input.shape
        Q = self.Q_weights(input)
        K = self.K_weights(input)
        V = self.V_weights(input)
        attention_scores = Q @ K.transpose(1, 2)
        attention_scores = attention_scores.masked_fill(
            self.causal_attention_mask[:tokens_num, :tokens_num] == 0, -torch.inf)
        attention_scores = attention_scores / (K.shape[-1] ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        return attention_scores @ V


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, their outputs are joined into one vector."""

    def __init__(self, config):
        """Create the heads and the linear layer after them."""
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config["heads_num"])])
        self.linear = nn.Linear(config["embedding_dim"], config["embedding_dim"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        """Return the joined output of all heads, shape (batch, tokens, embedding_dim)."""
        heads_outputs = [head(input) for head in self.heads]
        scores_change = self.linear(torch.cat(heads_outputs, dim=-1))
        return self.dropout(scores_change)


class FeedForward(nn.Module):
    """Two linear layers with GELU, applied to every character on its own."""

    def __init__(self, config):
        """Create two linear layers with a middle layer four times as wide as the embedding."""
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["embedding_dim"] * 4),
            nn.GELU(),
            nn.Linear(config["embedding_dim"] * 4, config["embedding_dim"]),
            nn.Dropout(config["dropout_rate"]),
        )

    def forward(self, input):
        """Return the output of the two layers."""
        return self.linear_layers(input)


class Block(nn.Module):
    """One Transformer block, attention and feed-forward part, each with a residual connection."""

    def __init__(self, config):
        """Create the attention, the feed-forward part and the two layer norms."""
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["embedding_dim"])
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["embedding_dim"])

    def forward(self, input):
        """Return the output of the block, same shape as the input."""
        x = input + self.multi_head(self.layer_norm_1(input))
        return x + self.feed_forward(self.layer_norm_2(x))


class NameGPT(nn.Module):
    """Decoder-only Transformer that predicts the next character of a company name."""

    def __init__(self, config):
        """Create the embeddings, the blocks, the final layer norm and the output layer."""
        super().__init__()
        embedding_dim = config["embedding_dim"]
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], embedding_dim)
        self.positional_embedding_layer = nn.Embedding(config["context_size"], embedding_dim)
        self.layers = nn.Sequential(*[Block(config) for _ in range(config["layers_num"])])
        self.layer_norm = nn.LayerNorm(config["embedding_dim"])
        self.unembedding = nn.Linear(config["embedding_dim"], config["vocabulary_size"], bias=False)

    def forward(self, token_ids):
        """Return the logits for the next token, shape (batch, tokens, vocabulary)."""
        batch_size, tokens_num = token_ids.shape
        x = self.token_embedding_layer(token_ids)
        positions = torch.arange(tokens_num, device=token_ids.device)
        x = x + self.positional_embedding_layer(positions)
        x = self.layers(x)
        x = self.layer_norm(x)
        return self.unembedding(x)
