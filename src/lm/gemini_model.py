# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
#
# from lm.utils import count_params
# import math
#
# """
# Dimension symbols:
#     B - batch size
#     S - sequence length
#     D - hidden dimension (n_embd)
#     H - number of attention heads (n_head)
#     HD - hidden dimension of a single attention head (d // n_head)
#     V - size of the vocabulary
# """
#
# class PositionalEncoding(nn.Module):
#     """The positional encoding in the model"""
#
#     def __init__(self, n_embd: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
#         pe = torch.zeros(1, max_len, n_embd)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         """
#         Args:
#             x: Tensor, shape (B, S, D)
#
#         Returns:
#             x + sinusoidal positional encoding
#         """
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)
#
# class MultiHeadAttention(nn.Module):
#     """The multi-head attention module in a decoder block."""
#
#     def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1):
#         super().__init__()
#         """Initialize the modules used by multi-head attention."""
#
#         self.n_head = n_head
#         attn_hidden_dim = n_embd // n_head
#
#         self.q_attn = nn.Linear(n_embd, n_embd)
#         self.k_attn = nn.Linear(n_embd, n_embd)
#         self.v_attn = nn.Linear(n_embd, n_embd)
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.dropout = nn.Dropout(p_dropout)
#
#         scale_factor = 1 / torch.sqrt(torch.tensor(attn_hidden_dim))
#         self.register_buffer("scale_factor", scale_factor)
#
#     def q_kT_v(
#         self, x: torch.FloatTensor
#     ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
#         """Project the hidden states to q, kT, v prior to computing attention.
#
#         Args:
#             x: embeddings or hidden states (B, S, D) from the previous decoder block
#
#         Returns:
#             q: The query vector used by multi-head attention (B, H, S, HD)
#             kT: The transpose of the key vector used by multi-head attention (B, H, HD, S)
#             v: The value vector used by multi-head attention (B, H, S, HD)
#
#         hint: torch.view
#         """
#         q = ...
#         kT = ...
#         v = ...
#
#         return q, kT, v
#
#     def self_attention(
#         self,
#         q: torch.FloatTensor,
#         kT: torch.FloatTensor,
#         v: torch.FloatTensor,
#         attention_mask: torch.FloatTensor | None = None,
#     ) -> torch.FloatTensor:
#         """Compute multi-head attention over the inputs.
#
#         Args:
#             q: The query vector used by multi-head attention (B, H, S, HD)
#             kT: The transpose of the key vector used by multi-head attention (B, H, HD, S)
#             v: The value vector used by multi-head attention (B, H, S, HD)
#             attention_mask (optional): Mask indicating tokens that shouldn't
#               be included in self-attention (B, S). 1 stands for a token that is
#               included, and 0 stands for a token that isn't.
#
#         Returns:
#             attn: Outputs of applying multi-head attention to the inputs (B, S, D)
#         """
#
#         # compute the attention weights using q and kT
#         qkT =...
#         unmasked_attn_logits = qkT * self.scale_factor
#
#         """
#         In decoder models, attention logits are masked such that computation at
#         each position does not involve embeddings / hidden states of future
#         positions.
#
#         This boolean mask should have shape (S, S) and has value True iff
#         position i is allowed to attend to position j (i.e., j <= i).
#
#         Example (S = 5):
#         causal_mask = tensor([
#          [ True, False, False, False, False],
#          [ True,  True, False, False, False],
#          [ True,  True,  True, False, False],
#          [ True,  True,  True,  True, False],
#          [ True,  True,  True,  True,  True]
#         ])
#
#         Note that `causal mask` needs to be on the same device as the input
#         tensors (q, kT, v). You can move a tensor to the right device by calling
#         `tensor.to(q.device)`.
#
#         Hint: torch.triu or torch.tril
#         """
#         causal_mask = ...
#
#         """
#         Sometimes, we want to pad the input sequences so that they have the same
#         length and can fit into the same batch. These padding tokens should not
#         have any effect on the output of self-attention. To achieve this, we
#         need to mask out the logits that correspond to those tokens.
#
#         Example (B = 2, S = 5):
#         causal_mask = tensor([
#          [ True, False, False, False, False],
#          [ True,  True, False, False, False],
#          [ True,  True,  True, False, False],
#          [ True,  True,  True,  True, False],
#          [ True,  True,  True,  True,  True]
#         ])
#
#         attention_mask = tensor([
#          [0., 0., 1., 1., 1.],
#          [1., 1., 1., 1., 1.]
#         ])
#
#         mask = tensor([
#         [[[False, False, False, False, False],
#           [False, False, False, False, False],
#           [False, False,  True, False, False],
#           [False, False,  True,  True, False],
#           [False, False,  True,  True,  True]]],
#
#         [[[ True, False, False, False, False],
#           [ True,  True, False, False, False],
#           [ True,  True,  True, False, False],
#           [ True,  True,  True,  True, False],
#           [ True,  True,  True,  True,  True]]]
#         ])
#
#         Note that `mask` needs to be on the same device as the input tensors
#         q, kT and v.
#         """
#
#         if attention_mask is None:
#             mask = causal_mask
#         else:
#             mask = ...
#
#         """
#         Fill unmasked_attn_logits with **float_min** wherever causal mask has value False.
#
#         Hint: torch.masked_fill
#         """
#         float_min = torch.finfo(q.dtype).min
#         attn_logits = ...
#         attn_weights = ...
#         attn_weights = self.dropout(attn_weights)
#
#         # scale value by the attention weights.
#         attn = ...
#
#         return attn
#
#     def projection(self, attn: torch.FloatTensor) -> torch.FloatTensor:
#         """Apply a dropout and a linear projection to outputs of attention"""
#         return self.dropout(self.proj(attn))
#
#     def forward(
#         self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None
#     ) -> torch.FloatTensor:
#         """A full forward pass of the multi-head attention module.
#
#         Args:
#             x: embeddings or hidden states (B, S, D) from the previous decoder block
#
#         Returns:
#             y: outputs (B, S, D) of the multi-head attention module
#         """
#
#         y = ...
#         return y
#
#
# class FeedForward(nn.Module):
#     """The feedforward attention module in a decoder block."""
#
#     def __init__(self, n_embd: int, p_dropout: float = 0.1):
#         """Initialize the modules used by feedforward."""
#         super().__init__()
#
#         middle_dim = 4 * n_embd  # stick to what GPT-2 does
#         self.linear_in = nn.Linear(n_embd, middle_dim)
#         self.linear_out = nn.Linear(middle_dim, n_embd)
#         self.dropout = nn.Dropout(p_dropout)
#
#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         """A full forward pass of the feedforward module.
#
#         Args:
#             x: outputs (B, S, D) of the first Add & Norm operation
#
#         Returns:
#             z: outputs (B, S, D) of the feedforward module
#
#         Different from what you saw in class which uses ReLU as the activation,
#         we are going to follow GPT-2 which uses GeLU. You should also apply
#         self.dropout to the output.
#         """
#
#         y = F.gelu(...)
#         z = self.dropout(...)
#         return z
#
#
# class DecoderBlock(nn.Module):
#     """A single decoder block in a decoder language model."""
#
#     def __init__(self, n_embd: int, n_head: int):
#         """Initialize the modules used in a decoder block."""
#         super().__init__()
#
#         self.ln_1 = nn.LayerNorm(n_embd)
#         self.mha = MultiHeadAttention(n_embd, n_head)
#         self.ff = FeedForward(n_embd)
#         self.ln_2 = nn.LayerNorm(n_embd)
#
#     def forward(
#         self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None
#     ) -> torch.FloatTensor:
#         """A full forward pass of the decoder block.
#
#         Args:
#             x: embeddings or hidden states (B, S, D) from the previous decoder block
#             attention_mask (optional): Mask indicating tokens that shouldn't
#               be included in self-attention (B, S). 1 stands for a token that is
#               included, and 0 stands for a token that isn't.
#         Returns:
#             y: outputs of the current decoder block
#
#         Different from what you saw in class which uses ReLU as the activation,
#         we are going to follow GPT-2 which uses GeLU. You should also apply
#         self.dropout to the output.
#
#         A note on where to place layer normalization (LN): in the lecture, you
#         saw "post-LN", which applies LN to the outputs of MHA / FF modules after
#         the residual is added. Another approach to do this is "pre-LN", which
#         appiles LN to the inputs of the attention and feedforward modules. Both
#         implementations should pass the tests. See explanations here:
#         https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab
#         """
#
#         return ...
#
#
# class DecoderLM(nn.Module):
#     """The decoder language model."""
#
#     def __init__(
#         self,
#         n_vocab: int,
#         n_embd: int,
#         n_head: int,
#         n_positions: int,
#         n_layer: int,
#         p_dropout: float = 0.1,
#     ):
#         super().__init__()
#
#         self.n_vocab = n_vocab
#         self.n_embd = n_embd
#         self.n_head = n_head
#         self.n_positions = n_positions
#         self.n_layer = n_layer
#         self.p_dropout = p_dropout
#
#         self.token_embeddings = nn.Embedding(n_vocab, n_embd)
#         self.position_embeddings = PositionalEncoding(n_embd)
#         self.blocks = nn.ModuleList(
#             [DecoderBlock(n_embd, n_head) for _ in range(n_layer)]
#         )
#         self.ln = nn.LayerNorm(n_embd)
#         self.dropout = nn.Dropout(self.p_dropout)
#
#         # initialize weights according to nanoGPT
#         self.apply(self._init_weights)
#         for pn, p in self.named_parameters():
#             if pn.endswith("c_proj.weight"):
#                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(2 * n_layer))
#
#         # count flops per token according to nanoGPT
#         self.flops_per_token = (
#             6 * count_params(self) + 12 * n_layer * n_embd * n_positions
#         )
#
#     def embed(
#         self,
#         input_ids: torch.LongTensor,
#         attention_mask: torch.FloatTensor | None = None,
#     ) -> torch.FloatTensor:
#         """Convert input_ids to embeddings (token_embeddings + positional_embeddings).
#
#         Args:
#             input_ids: tokens ids with shape (B, S)
#             attention_mask (optional): Mask indicating whether tokens should be
#               ignored.
#
#         Returns:
#             embeddings: token representations with shape (B, S, D)
#         """
#
#         """
#         Position ids are indices of tokens in the sequence. When attention_mask
#         isn't provided, they are simply [0, 1, 2, ...] for every sequence in the
#         batch. When they are provided, you should ignore tokens with attention_mask
#         equal to 0.
#
#         Example (B = 2, S = 5):
#
#         attention_mask = tensor([
#          [0., 0., 1., 1., 1.],
#          [1., 1., 1., 1., 1.]
#         ])
#
#         position_ids = tensor([
#          [0, 0, 0, 1, 2],
#          [0, 1, 2, 3, 4]
#         ])
#
#         Note that the position ids for masked out tokens do not matter, as long
#         as they don't trigger out-of-bounds errors when fed into the embedding
#         layer. I.e., they should be within [0, n_positions).
#
#         Hint: torch.cumsum
#         """
#
#         assert input_ids.shape[1] <= self.n_positions
#         token_embeddings = ...
#         positional_embeddings = ...
#         return self.dropout(token_embeddings + positional_embeddings)
#
#     def token_logits(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         """Project the final hidden states of the model to token logits.
#
#         Args:
#             x: hidden states produced by the final decoder block (B, S, D)
#
#         Returns:
#             logits: logits corresponding to the predicted next token likelihoods (B, S, V)
#
#         Hint: Question 1.2.
#         """
#
#         logits = ...
#         return logits
#
#     def forward(
#         self,
#         input_ids: torch.LongTensor,
#         attention_mask: torch.FloatTensor | None = None,
#     ) -> torch.FloatTensor:
#         """A forward pass of the decoder LM, converting input_ids to token logits.
#
#         Args:
#             input_ids: tokens ids with shape (B, S)
#             attention_mask (optional): Mask indicating whether tokens should be
#               ignored.
#
#         Returns:
#             logits: logits corresponding to the predicted next token likelihoods (B, S, V)
#         """
#
#         logits = ...
#         return logits
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not ...:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lm.utils import count_params
import math

"""
Dimension symbols:
    B - batch size
    S - sequence length
    D - hidden dimension (n_embd)
    H - number of attention heads (n_head)
    HD - hidden dimension of a single attention head (d // n_head)
    V - size of the vocabulary
"""


class PositionalEncoding(nn.Module):
    """The positional encoding in the model"""

    def __init__(self, n_embd: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(1, max_len, n_embd)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: Tensor, shape (B, S, D)

        Returns:
            x + sinusoidal positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """The multi-head attention module in a decoder block."""

    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1):
        super().__init__()
        """Initialize the modules used by multi-head attention."""

        self.n_head = n_head
        attn_hidden_dim = n_embd // n_head

        self.q_attn = nn.Linear(n_embd, n_embd)
        self.k_attn = nn.Linear(n_embd, n_embd)
        self.v_attn = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p_dropout)

        scale_factor = 1 / torch.sqrt(torch.tensor(attn_hidden_dim))
        self.register_buffer("scale_factor", scale_factor)

    def q_kT_v(
            self, x: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Project the hidden states to q, kT, v prior to computing attention.

        Args:
            x: embeddings or hidden states (B, S, D) from the previous decoder block

        Returns:
            q: The query vector used by multi-head attention (B, H, S, HD)
            kT: The transpose of the key vector used by multi-head attention (B, H, HD, S)
            v: The value vector used by multi-head attention (B, H, S, HD)

        hint: torch.view or einops.rearrange
        """
        # Project input x to q, k, and v.
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)

        # Reshape for multi-head attention.
        # (B, S, D) -> (B, S, H, HD) -> (B, H, S, HD)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_head)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_head)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_head)

        # Transpose k for matrix multiplication.
        kT = rearrange(k, 'b h s d -> b h d s')

        return q, kT, v

    def self_attention(
            self,
            q: torch.FloatTensor,
            kT: torch.FloatTensor,
            v: torch.FloatTensor,
            attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Compute multi-head attention over the inputs.

        Args:
            q: The query vector used by multi-head attention (B, H, S, HD)
            kT: The transpose of the key vector used by multi-head attention (B, H, HD, S)
            v: The value vector used by multi-head attention (B, H, S, HD)
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B, S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.

        Returns:
            attn: Outputs of applying multi-head attention to the inputs (B, S, D)
        """

        # compute the attention weights using q and kT
        qkT = torch.matmul(q, kT)
        unmasked_attn_logits = qkT * self.scale_factor

        S = q.size(-2)

        """
        In decoder models, attention logits are masked such that computation at
        each position does not involve embeddings / hidden states of future
        positions.

        This boolean mask should have shape (S, S) and has value True iff
        position i is allowed to attend to position j (i.e., j <= i).

        Example (S = 5):
        causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])

        Note that `causal mask` needs to be on the same device as the input
        tensors (q, kT, v). You can move a tensor to the right device by calling
        `tensor.to(q.device)`.

        Hint: torch.tril
        """
        causal_mask = torch.tril(torch.ones(S, S, device=q.device)).bool()

        """
        Sometimes, we want to pad the input sequences so that they have the same
        length and can fit into the same batch. These padding tokens should not
        have any effect on the output of self-attention. To achieve this, we
        need to mask out the logits that correspond to those tokens.

        Example (B = 2, S = 5):
        causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])

        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        mask = tensor([
        [[[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False,  True, False, False],
          [False, False,  True,  True, False],
          [False, False,  True,  True,  True]]],

        [[[ True, False, False, False, False],
          [ True,  True, False, False, False],
          [ True,  True,  True, False, False],
          [ True,  True,  True,  True, False],
          [ True,  True,  True,  True,  True]]]
        ])

        Note that `mask` needs to be on the same device as the input tensors
        q, kT and v.
        """

        if attention_mask is None:
            mask = causal_mask
        else:
            # Combine causal mask with the padding mask.
            # attention_mask is (B, S), we need to expand it for broadcasting.
            padding_mask = attention_mask.view(-1, 1, 1, S)
            mask = causal_mask & padding_mask.bool()
        """
        Fill unmasked_attn_logits with **float_min** wherever causal mask has value False.

        Hint: torch.masked_fill
        """
        float_min = torch.finfo(q.dtype).min
        attn_logits = unmasked_attn_logits.masked_fill(mask == 0, float_min)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # scale value by the attention weights.
        attn_vectors = torch.matmul(attn_weights, v)

        # Reshape back to (B, S, D)
        attn = rearrange(attn_vectors, 'b h s d -> b s (h d)')

        return attn

    def projection(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        """Apply a dropout and a linear projection to outputs of attention"""
        return self.dropout(self.proj(attn))

    def forward(
            self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None
    ) -> torch.FloatTensor:
        """A full forward pass of the multi-head attention module.

        Args:
            x: embeddings or hidden states (B, S, D) from the previous decoder block

        Returns:
            y: outputs (B, S, D) of the multi-head attention module
        """
        q, kT, v = self.q_kT_v(x)
        attn = self.self_attention(q, kT, v, attention_mask)
        y = self.projection(attn)
        return y


class FeedForward(nn.Module):
    """The feedforward attention module in a decoder block."""

    def __init__(self, n_embd: int, p_dropout: float = 0.1):
        """Initialize the modules used by feedforward."""
        super().__init__()

        middle_dim = 4 * n_embd  # stick to what GPT-2 does
        self.linear_in = nn.Linear(n_embd, middle_dim)
        self.linear_out = nn.Linear(middle_dim, n_embd)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """A full forward pass of the feedforward module.

        Args:
            x: outputs (B, S, D) of the first Add & Norm operation

        Returns:
            z: outputs (B, S, D) of the feedforward module

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.
        """

        hidden = self.linear_in(x)
        y = F.gelu(hidden)
        z = self.dropout(self.linear_out(y))
        return z


class DecoderBlock(nn.Module):
    """A single decoder block in a decoder language model."""

    def __init__(self, n_embd: int, n_head: int):
        """Initialize the modules used in a decoder block."""
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(
            self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None
    ) -> torch.FloatTensor:
        """A full forward pass of the decoder block.

        Args:
            x: embeddings or hidden states (B, S, D) from the previous decoder block
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B, S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.
        Returns:
            y: outputs of the current decoder block

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.

        A note on where to place layer normalization (LN): in the lecture, you
        saw "post-LN", which applies LN to the outputs of MHA / FF modules after
        the residual is added. Another approach to do this is "pre-LN", which
        appiles LN to the inputs of the attention and feedforward modules. Both
        implementations should pass the tests. See explanations here:
        https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab
        """
        # First residual connection with multi-head attention
        attn_output = self.mha(self.ln_1(x), attention_mask)
        x = x + attn_output

        # Second residual connection with feed-forward network
        ff_output = self.ff(self.ln_2(x))
        y = x + ff_output

        return y


class DecoderLM(nn.Module):
    """The decoder language model."""

    def __init__(
            self,
            n_vocab: int,
            n_embd: int,
            n_head: int,
            n_positions: int,
            n_layer: int,
            p_dropout: float = 0.1,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.p_dropout = p_dropout

        self.token_embeddings = nn.Embedding(n_vocab, n_embd)
        self.position_embeddings = PositionalEncoding(n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.p_dropout)

        # initialize weights according to nanoGPT
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(2 * n_layer))

        # count flops per token according to nanoGPT
        self.flops_per_token = (
                6 * count_params(self) + 12 * n_layer * n_embd * n_positions
        )

    def embed(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Convert input_ids to embeddings (token_embeddings + positional_embeddings).

        Args:
            input_ids: tokens ids with shape (B, S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            embeddings: token representations with shape (B, S, D)
        """

        """
        Position ids are indices of tokens in the sequence. When attention_mask
        isn't provided, they are simply [0, 1, 2, ...] for every sequence in the
        batch. When they are provided, you should ignore tokens with attention_mask
        equal to 0.

        Example (B = 2, S = 5):

        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        position_ids = tensor([
         [0, 0, 0, 1, 2],
         [0, 1, 2, 3, 4]
        ])

        Note that the position ids for masked out tokens do not matter, as long
        as they don't trigger out-of-bounds errors when fed into the embedding
        layer. I.e., they should be within [0, n_positions).

        Hint: torch.cumsum
        """

        assert input_ids.shape[1] <= self.n_positions
        token_embeddings = self.token_embeddings(input_ids)
        # PositionalEncoding class handles adding positional info
        return self.position_embeddings(token_embeddings)

    def token_logits(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Project the final hidden states of the model to token logits.

        Args:
            x: hidden states produced by the final decoder block (B, S, D)

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B, S, V)

        Hint: Question 1.2.
        """
        # Project hidden states to vocabulary size.
        # The weight matrix of token_embeddings is (V, D). We need to transpose it.
        logits = F.linear(x, self.token_embeddings.weight)
        return logits

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """A forward pass of the decoder LM, converting input_ids to token logits.

        Args:
            input_ids: tokens ids with shape (B, S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B, S, V)
        """

        # 1. Get embeddings
        x = self.embed(input_ids, attention_mask)

        # 2. Pass through decoder blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # 3. Final layer norm
        x = self.ln(x)

        # 4. Get logits
        logits = self.token_logits(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
