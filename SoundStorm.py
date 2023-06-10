import math
import random

import numpy as np
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, EinMix
from torch import nn
import torch.nn.functional as F
from core.conformer import Conformer


class SoundStorm(nn.Module):

    def __init__(self, dim=1024, heads=16, linear_units=4096, num_blocks=12, semantic_codebook_size=1024,
                semantic_num_quantizers=1, acoustic_codebook_size=1024, acoustic_num_quantizers=8,
                positionwise_conv_kernel_size=5):

        super().__init__()
        num_codes_with_mask = acoustic_codebook_size + 1

        self.semantic_embeds = nn.Embedding((semantic_codebook_size + 1) * semantic_num_quantizers, dim)

        self.code_embeds = nn.Embedding(num_codes_with_mask * acoustic_num_quantizers, dim)

        self.register_buffer('quantizer_offsets', torch.arange(acoustic_num_quantizers) * num_codes_with_mask,
                             persistent=False)

        self.register_buffer('mask_token_id', self.quantizer_offsets + num_codes_with_mask, persistent=False)

        self.register_buffer('sos_tokens', self.mask_token_id + 1, persistent=False)

        self.lm = Conformer(
            attention_dim=dim,
            attention_heads=heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size
        )

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * acoustic_num_quantizers),
            Rearrange('b n (h d) -> b (n h) d', h=acoustic_num_quantizers)
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b (n q) d -> b n q d', q=acoustic_num_quantizers),
            EinMix(
                'b n q d -> b n q l',
                weight_shape='q d l',
                bias_shape='q l',
                q=acoustic_num_quantizers,
                l=acoustic_codebook_size,
                d=dim
            )
        )

    def masking(self, codes, q=None, t=None):

        codes = rearrange(codes, 'b n q -> q b n')
        q = random.randint(0, codes.size(0) - 1) if q is None else q
        t = random.randint(0, codes.shape[-1] - 1) if t is None else t
        t_mask = torch.ones(codes.shape)
        t_mask[:, :, t:] = 0
        t_mask[0:q] = 1

        masked_indices = self.mask_token_id[q-1] * torch.ones_like(codes, device=codes.device)
        codes = t_mask * codes + (1 - t_mask) * masked_indices

        indices = codes[q - 1]

        gamma = self.gamma_func()
        gammas = gamma(np.random.uniform())
        r = math.floor(gammas * indices.shape[1])
        sample = torch.rand(indices.shape, device=indices.device).topk(r, dim=1).indices
        mask = torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
        mask.scatter_(dim=1, index=sample, value=True)
        mask[:, :t] = True
        masked_indices = self.mask_token_id[q-1] * torch.ones_like(indices, device=indices.device)
        codes[q - 1] = mask * indices + (~mask) * masked_indices

        codes = rearrange(codes, 'q b n -> b n q')

        return codes, mask, q  # [B, Q, N+1]


    def forward(self, cond, codes, return_loss=True):
        """
        cond: [B, Len]
        codes: [B, N_q, Len]
        """

        b, q, n = codes.shape

        codes = rearrange(codes, 'b q n -> b n q', q=q)
        orig_codes = codes.clone()
        codes = codes + self.quantizer_offsets

        emb, masks, q = self.masking(codes)

        emb = self.code_embeds(emb.long())                     # [B, n, q, d]

        semb = self.semantic_embeds(cond)               # [B, n, d]

        emb = reduce(emb, 'b n q d -> b n d', 'sum')                  # [B, n, d]

        emb = emb + semb

        out, _ = self.lm(emb, None)                            # [B, n, d]

        out = self.heads(out)                         # [B, q*n, d]

        logits = self.to_logits(out)                  # [B, n, q, d]

        if return_loss:
            logits = logits[:, :, q-1].squeeze(2)       # [B, n, d]
            orig_codes = orig_codes[:, :, q-1]          # [B, n]

            loss = F.cross_entropy(
                logits[masks],
                orig_codes[masks]
            )
            return loss, logits, out
        return logits, out, masks

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError


def num_params(model, print_out=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print("Trainable Parameters: %.3fM" % parameters)


if __name__ == '__main__':

    cond = torch.ones([2, 100]).long()
    codes = torch.ones([2, 8, 100]).long()

    model = SoundStorm()

    num_params(model)


    logits, out, mask = model(cond, codes)


