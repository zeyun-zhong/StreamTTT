"""
The fast-weight memory follows the closed-form chunk-wise formulation of E²-TTT (arXiv:2608.21308),
building on the large-chunk TTT of LaCT (arXiv:2505.23884)
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from typing import Optional

from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNormGated

# Optional fused CUDA kernel. The `try` block also keeps transformers'
# `check_imports` from making `causal_conv1d` a hard requirement when this
# file is loaded as remote code from the Hub.
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def layernorm_fwd(x_f32: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-8):
    """
    x_f32: [B, D, L]    (float32 compute recommended)
    gamma: [B, D, 1]
    beta:  [B, D, 1]
    returns:
      y:       [B, D, L]  (cast to gamma.dtype)
      mean:    [B, 1, L]
      inv_std: [B, 1, L]
    """
    out_dtype = gamma.dtype
    mu = x_f32.mean(dim=1, keepdim=True)
    xc = x_f32 - mu
    var = (xc * xc).mean(dim=1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    xhat = xc * inv_std
    y = xhat * gamma.float() + beta.float()  # fp32
    return y.to(out_dtype), mu, inv_std


# Backward: only dpre (input grad), using cached mean & inv_std
def layernorm_bwd(x32: torch.Tensor, dY: torch.Tensor, gamma: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor):
    """
    x32:   [B, D, L] (same tensor used in forward, cast to float32)
    dY:    [B, D, L] (grad wrt output y)
    gamma: [B, D, 1] (detached when used here)
    mean:  [B, 1, L] (cached)
    inv_std: [B, 1, L] (cached)
    returns:
      dpre: [B, D, L] (cast to gamma.dtype)
    """
    B, D, L = x32.shape
    dY32 = dY.float()
    g32 = gamma.float()

    xhat = (x32 - mean) * inv_std                                # [B,D,L]

    gh = dY32 * g32                                              # [B,D,L]
    sum_gh = gh.sum(dim=1, keepdim=True)                         # [B,1,L]
    sum_gh_xhat = (gh * xhat).sum(dim=1, keepdim=True)           # [B,1,L]

    dpre = (gh - (sum_gh + xhat * sum_gh_xhat) / D) * inv_std    # [B,D,L]
    return dpre.to(gamma.dtype)


def build_token_weights_chunk_log(log_beta, log_gamma):
    """
    log_beta, log_gamma: [B, L, D] (log-space; typically <= 0)
    Returns (all broadcastable over [B, L, D] unless noted):
      log_alpha : [B, L, D]    where alpha = exp(log_alpha)
      sbeta     : [B, L, D]    = exp( sum_{j=t+1..L-1} log_beta_j )
      carry_u   : [B, 1, D]    = exp( sum_{j=0..L-1} log_gamma_j )
      carry_m   : [B, 1, D]    = exp( sum log_beta ) * sum_{q} exp( sum_{k=q+1..} (log_gamma - log_beta) )
    """
    # ---- make shapes [B, L] ----
    assert log_beta.dim() == 3 and log_gamma.dim() == 3
    log_beta = log_beta.float()
    log_gamma = log_gamma.float()

    # suffix log-products: log_sx[t] = Σ_{j=t+1..} log x_j
    # Implement via cumsum on reversed, padding a trailing 0 (exp(0)=1 sentinel)
    def suffix_logprod(logx):                              # logx: [B, L]
        pad  = torch.nn.functional.pad(logx, (0, 0, 0, 1), value=0.0)
        suf  = torch.flip(torch.cumsum(torch.flip(pad, [1]), dim=1), [1])
        return suf[:, 1:], suf[:, :1]

    log_sbeta,  log_prodb  = suffix_logprod(log_beta)     # [B, L], [B, 1]
    log_sgamma, log_prodg  = suffix_logprod(log_gamma)    # [B, L], [B, 1]

    # r_t = sγ_t / sbeta_t  => log_r = log_sgamma - log_sbeta
    log_r = log_sgamma - log_sbeta                        # [B, L]

    # R_t = Σ_{q=t..} r_q = suffix-sum(exp(log_r))
    # Use logcumsumexp on the reversed axis (loop-free, stable)
    R_log = torch.flip(torch.logcumsumexp(torch.flip(log_r, [1]), dim=1), [1])  # [B, L]

    # final weights
    sbeta   = torch.exp(log_sbeta)
    log_alpha = log_sbeta + R_log
    carry_u = torch.exp(log_prodg)
    carry_m = (log_prodb + R_log[:, :1]).exp()

    return log_alpha, sbeta, carry_u, carry_m


def build_token_weights_chunk_log_nomomentum(log_gamma):  # [B,L,d]
    assert log_gamma.dim() == 3
    log_gamma = log_gamma.float()
    pad  = torch.nn.functional.pad(log_gamma, (0, 0, 0, 1), value=0.0)
    suf = torch.flip(torch.cumsum(torch.flip(pad, [1]), dim=1), [1])
    log_sgamma, log_prodg = suf[:, 1:], suf[:, :1]
    alpha_log = log_sgamma
    carry_u = torch.exp(log_prodg)
    return alpha_log, carry_u


def causal_conv1d(x, weight, bias=None, activation=None):
    """Depthwise causal conv1d, matching `causal_conv1d_fn`'s semantics.

    Uses the fused CUDA kernel when `causal-conv1d` is installed and the input is
    on GPU; falls back to an equivalent (slower) PyTorch implementation otherwise,
    so the model also runs without the kernel and on CPU.

    Args:
        x: (batch, channels, seqlen)
        weight: (channels, kernel_size)
        bias: (channels,) or None
        activation: None, "silu" or "swish"
    Returns:
        (batch, channels, seqlen)
    """
    if causal_conv1d_fn is not None and x.is_cuda:
        return causal_conv1d_fn(x=x, weight=weight, bias=bias, activation=activation, seq_idx=None)

    if activation not in (None, "silu", "swish"):
        raise NotImplementedError(f"causal_conv1d activation {activation!r} is not supported")

    kernel_size = weight.shape[-1]
    out = F.conv1d(
        F.pad(x, (kernel_size - 1, 0)),
        weight.unsqueeze(1).to(dtype=x.dtype),
        bias=None if bias is None else bias.to(dtype=x.dtype),
        groups=x.shape[1],
    )
    return out if activation is None else F.silu(out)


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


def pow2_ceil(n: int) -> int:
    if n <= 1:
        return max(n, 1)   # 0→1, 1→1
    return 1 << ((n - 1).bit_length())


def pick_tail_len_pow2(score, value_min=1e-9, window_min=64):
    seq_len = score.shape[1]
    mask = score < value_min
    num_prune = torch.cumprod(mask, dim=1).sum(dim=1).min().item()
    window_keep = seq_len - num_prune
    if window_keep > 0:
        window_keep = pow2_ceil(window_keep)  # round down to power of two
        window_keep = max(window_keep, window_min)
    return window_keep


def gradient_clip(grad, max_norm=1.0):
    assert grad.dim() == 3
    cur_norm = grad.norm(p=2, dim=(1,2), keepdim=True)
    if (cur_norm > max_norm).any():
        scale = (max_norm / (cur_norm + 1e-12)).clamp(max=1.0)
        grad = grad * scale
    return grad


class FastWeightBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size = getattr(config, "lact_chunk_size", 2048)
        self.num_fw_heads = getattr(config, "num_fw_heads", 4)
        self.num_fw_kv_heads = getattr(config, "num_fw_kv_heads", 4)
        assert self.num_fw_heads == self.num_fw_kv_heads  # let's keep it simple here

        self.fw_head_dim = config.hidden_size // self.num_fw_heads
        dim = config.hidden_size + (self.fw_head_dim * self.num_fw_kv_heads) * 2
        self.qkv = nn.Linear(config.hidden_size, dim, bias=False)

        self.inter_multi = float(getattr(config, "inter_multi", 1.0))
        self.lr_parameterization = str(getattr(config, "lr_parameterization", "mamba"))
        self.use_residual = getattr(config, "use_residual", False)  # default to False,
        assert self.use_residual

        d_in = d_out = self.fw_head_dim
        d_h = int(d_in * self.inter_multi)
        self.d_h = d_h

        self.w0 = nn.Parameter(torch.randn(self.num_fw_kv_heads, d_h, d_in))
        self.w2 = nn.Parameter(torch.randn(self.num_fw_kv_heads, d_h, d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_fw_kv_heads, d_out, d_h))

        self.activation = config.hidden_act
        self.conv_dim = config.hidden_size + (self.fw_head_dim * self.num_fw_kv_heads) * 2
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=4,
            groups=self.conv_dim,
            padding=3,
        )
        self.conv_state_len = self.conv1d.kernel_size[0] - 1
        # 1e-5 for w/o momentum; 5e-7 for w/ momentum
        self.base_lr = getattr(config, "ttt_base_lr", 0.01)
        self.base_decay = getattr(config, "ttt_base_decay", 0.1)

        # We might need to think a better lr approach
        if self.lr_parameterization.lower() == "mamba":
            self.base_lr_inv = inv_softplus(self.base_lr)

        lr_outg_dim = self.num_fw_kv_heads + self.fw_head_dim
        self.lr_outg = nn.Linear(config.hidden_size, lr_outg_dim, bias=False)

        # momentum
        self.ttt_momentum = getattr(config, "ttt_momentum", "headwise")
        assert self.ttt_momentum in ["none", "headwise", "channelwise"]
        if self.ttt_momentum != "none":
            self.dim_momentum = self.num_fw_kv_heads if self.ttt_momentum == "headwise" else self.num_fw_kv_heads * self.fw_head_dim
            self.momentum_proj = nn.Linear(config.hidden_size, self.dim_momentum, bias=True)

        # weight decay
        self.weight_decay_type = config.ttt_weight_decay
        assert self.weight_decay_type in ["none", "headwise", "channelwise"]
        if self.weight_decay_type != "none":
            self.dim_decay = self.num_fw_kv_heads if self.weight_decay_type == "headwise" else self.num_fw_kv_heads * self.fw_head_dim
            self.decay_proj = nn.Linear(config.hidden_size, self.dim_decay, bias=False)

        # norm
        self.norm = Qwen3NextRMSNormGated(self.fw_head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # RMSNorm in the TTT update
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.num_fw_kv_heads, self.fw_head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.num_fw_kv_heads, self.fw_head_dim))

        self.verbose = getattr(config, "print_verbose", False)

    def _split_prev_states(self, prev_states):
        if prev_states is None:
            return None, None
        if len(prev_states) == 12:
            return prev_states[0], prev_states[1:]
        return None, prev_states

    def _apply_causal_conv(self, mixed_qkv: torch.Tensor, conv_prefix: Optional[torch.Tensor]):
        if conv_prefix is not None and conv_prefix.shape[-1] > 0:
            conv_input = torch.cat([conv_prefix.to(dtype=mixed_qkv.dtype, device=mixed_qkv.device), mixed_qkv], dim=-1)
            prefix_len = conv_prefix.shape[-1]
        else:
            conv_input = mixed_qkv
            prefix_len = 0

        conv_output = causal_conv1d(
            x=conv_input,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
        )
        if prefix_len > 0:
            conv_output = conv_output[..., prefix_len:]

        next_conv_prefix_len = min(self.conv_state_len, conv_input.shape[-1])
        next_conv_prefix = conv_input[..., -next_conv_prefix_len:].contiguous()
        return conv_output, next_conv_prefix

    def _init_weights(self):
        nn.init.ones_(self.ttt_norm_weight)
        nn.init.zeros_(self.ttt_norm_bias)
        nn.init.normal_(self.w0, mean=0.0, std=1.0 / math.sqrt(self.fw_head_dim))
        nn.init.normal_(self.w2, mean=0.0, std=1.0 / math.sqrt(self.fw_head_dim))
        nn.init.normal_(self.w1, mean=0.0, std=1.0 / math.sqrt(self.d_h))
        self.qkv.weight.data.normal_(mean=0.0, std=0.02)
        self.o_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.lr_outg.weight.data.normal_(mean=0.0, std=0.02)
        self.conv1d.weight.data.normal_(mean=0.0, std=0.02)
        if hasattr(self, "momentum_proj"):
            self.momentum_proj.weight.data.normal_(mean=0.0, std=0.02)
            if self.momentum_proj.bias is not None:
                self.momentum_proj.bias.data.zero_()
        if hasattr(self, "decay_proj"):
            self.decay_proj.weight.data.normal_(mean=0.0, std=0.02)
            if self.decay_proj.bias is not None:
                self.decay_proj.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,  # [b, s, d]
        prev_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        is_generation: bool = False,
    ):
        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        conv_prefix, prev_states = self._split_prev_states(prev_states)
        mixed_qkv, conv_prefix = self._apply_causal_conv(mixed_qkv, conv_prefix)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        q_dim, k_dim = self.fw_head_dim * self.num_fw_heads, self.fw_head_dim * self.num_fw_kv_heads
        q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, k_dim], dim=-1)

        # split into heads (batch-folded)
        q = rearrange(q, 'b s (nh d) -> (b nh) s d', nh=self.num_fw_heads)
        k = rearrange(k, 'b s (nh d) -> (b nh) s d', nh=self.num_fw_kv_heads)
        v = rearrange(v, 'b s (nh d) -> (b nh) s d', nh=self.num_fw_kv_heads)

        # L2 Norm, following most linear attention papers and ttt-video-dit
        eps = 1e-8
        q = F.normalize(q, p=2, dim=-1, eps=eps)
        k = F.normalize(k, p=2, dim=-1, eps=eps)

        v = self.ln_reconstruction_target(v, k)

        # Prepare lr, momentum, decay, and output_gate
        mixed_lr_outg = self.lr_outg(hidden_states)
        lr, outg = torch.split(
            mixed_lr_outg,
            [self.num_fw_kv_heads, self.fw_head_dim],
            dim=-1,
        )

        if self.lr_parameterization.lower() == "mamba":
            lr = F.softplus(lr.float() + self.base_lr_inv)
        elif self.lr_parameterization.lower() == "ttt":
            lr = F.sigmoid(lr.float()) * self.base_lr
        else:
            raise NotImplementedError(f"LR parameterization {self.lr_parameterization} not implemented")

        lr = rearrange(lr, 'b s (nh d) -> (b nh) s d', nh=self.num_fw_kv_heads)

        if self.ttt_momentum != "none":
            momentum_projected = self.momentum_proj(hidden_states)
            momentum = rearrange(momentum_projected, 'b s (nh d) -> (b nh) s d', nh=self.num_fw_kv_heads)
            log_momentum = F.logsigmoid(momentum.float()) / 16  # we will use .exp() later, use logspace for numerical stability
        else:
            log_momentum = torch.zeros_like(lr)  # placeholder, not used

        # This is optional weight decay for updating memory
        if self.weight_decay_type != "none":
            decay_projected = self.decay_proj(hidden_states)
            alpha = F.sigmoid(decay_projected.float()) * self.base_decay
            alpha = rearrange(alpha, 'b s (nh d) -> (b nh) s d', nh=self.num_fw_kv_heads)
            log_decay = torch.log1p(-lr * alpha)
        else:
            log_decay = torch.zeros_like(lr).float()  # logscale here, log(1.) == 0

        # We downscale the learning rate by chunk size
        lr = lr * (1. / self.chunk_size)

        update_fn = self.update_and_retrieve_memory_generate if is_generation else self.update_and_retrieve_memory

        fw_x, recurrent_state = update_fn(q, k, v, lr, log_decay, log_momentum, prev_states)
        recurrent_state = (conv_prefix, *recurrent_state)

        ttt_x = self.norm(fw_x, outg)
        ttt_x = rearrange(ttt_x, '(b nh) s d -> b s (nh d)', nh=self.num_fw_heads)
        ttt_x = self.o_proj(ttt_x)

        return ttt_x, recurrent_state

    def ln_reconstruction_target(self, XV, XK):
        # adapted from https://github.com/test-time-training/ttt-video-dit/blob/main/ttt/models/ssm/ttt_layer.py#L220C5-L235C23
        XV = XV - XK
        eps = 1e-8

        # We compute in FP32
        XV = F.layer_norm(XV.float(), normalized_shape=(XV.size(-1),), eps=eps)  # z-score

        # Apply per-head weight and bias.
        # self.ttt_norm_weight and self.ttt_norm_bias have shape [num_heads, head_dim].
        # We unsqueeze to make them broadcastable with XV_norm which is [B*num_heads, L, head_dim].
        XV = self.ttt_norm_weight.unsqueeze(1).float() * XV + self.ttt_norm_bias.unsqueeze(1).float()

        return XV.to(XK.dtype)

    def update_chunk_wo_momentum(
        self, q_chunk, k_chunk, v_chunk,  # [b, d, l]
        lr_chunk, log_decay_chunk, log_beta_chunk,  # [b, l, 1]
        w0, w1, w2,  # w0 & w2: [b, dh, d]; w1: [b, d, dh]
        m_init_w0, m_init_w1, m_init_w2, # not used here
        w0_norm, w1_norm, w2_norm
    ):
        # For RMSNorm
        ttt_norm_weight = self.ttt_norm_weight.unsqueeze(-1)
        ttt_norm_bias = self.ttt_norm_bias.unsqueeze(-1)

        # Retrieve memory (Read out)
        # layernorm + residual will be done once after all chunks are processed
        h = torch.bmm(w2, q_chunk)
        gate = F.silu(torch.bmm(w0, q_chunk))
        out = torch.bmm(w1, gate * h)

        # Forward
        # MLPs
        Z0 = torch.bmm(w0, k_chunk)  # [b, dh, l]
        Z2 = torch.bmm(w2, k_chunk)  # [b, dh, l]
        G = F.silu(Z0)  # [b, dh, l]
        H = G * Z2  # [b, dh, l]
        pre = torch.bmm(w1, H).float()  # we use residual here  [b, d, l]
        # RMSNorm + Residual (Residual in already included in the v_chunk, see self.ln_restruction_target)
        O, mu, inv_s = layernorm_fwd(pre, ttt_norm_weight, ttt_norm_bias)

        # MSE loss
        Err = O - v_chunk  # [b, d, l]

        log_alpha, carry_u = build_token_weights_chunk_log_nomomentum(log_decay_chunk)
        carry_u = carry_u.to(k_chunk.dtype)
        alpha = torch.exp(log_alpha)

        alpha_lr = (alpha * lr_chunk).to(k_chunk.dtype)
        Err_alpha = Err * alpha_lr.transpose(1, 2)

        # LayerNorm backward
        grad_pre = layernorm_bwd(pre, Err_alpha, ttt_norm_weight, mu, inv_s)

        # Backprop to hidden terms
        B = torch.bmm(w1.transpose(1, 2), grad_pre)  # [b, dh, l]
        sig = torch.sigmoid(Z0)
        silu_prime = sig * (1.0 + Z0 * (1.0 - sig))  # [b, dh, l]
        dZ2 = B * G  # [b, dh, l]
        dZ0 = B * Z2 * silu_prime  # [b, dh, l]

        # Calculating dw1
        dw1 = -torch.bmm(grad_pre, H.transpose(1, 2))

        # Calculating dw0 & dw2
        k_t = k_chunk.transpose(1, 2)  # [b, l, d]
        dw2 = -torch.bmm(dZ2, k_t)
        dw0 = -torch.bmm(dZ0, k_t)

        if self.verbose:
            if torch.isnan(dw0).any() or (dw0 > 1e6).any():
                print(alpha[0])
                print(carry_u)
                print(torch.cumsum(log_decay_chunk, dim=1).exp())
                print(log_decay_chunk.exp().cumprod(dim=1)[-1, :, 0].tolist())
                print(log_decay_chunk.exp()[-1, :, 0].tolist())
                print(f"Err mean: {Err.mean()} Err norm: {Err.norm()}")
                print(f"Err_alpha mean: {Err_alpha.mean()} Err_alpha norm: {Err_alpha.norm()}")
                raise ValueError

            rel_step = dw0.norm() / w0.norm()
            print(f"before clipping rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
            print(f"carry_u: {carry_u[:, 0, 0].tolist()}")
            print(f"dw0 head norm before clip {dw0.norm(dim=(1,2))}")

        # gradient clip
        dw0 = gradient_clip(dw0)
        dw1 = gradient_clip(dw1)
        dw2 = gradient_clip(dw2)

        # We now calculate the weights for the last token in the chunk
        upd_w0 = carry_u * w0 + dw0
        upd_w1 = carry_u.transpose(1, 2) * w1 + dw1
        upd_w2 = carry_u * w2 + dw2

        if self.verbose:
            rel_step = dw0.norm() / w0.norm()
            print(f"rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
            print(f"dw0 head norm after clip {dw0.norm(dim=(1, 2))}")
            print(f"MSE: {Err.pow(2).mean()}")

        # Weight normalization (same as in test training done right)
        upd_w0 = upd_w0 / (upd_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        upd_w1 = upd_w1 / (upd_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        upd_w2 = upd_w2 / (upd_w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        return out, upd_w0, upd_w1, upd_w2, m_init_w0, m_init_w1, m_init_w2


    def update_chunk(
        self, q_chunk, k_chunk, v_chunk,  # [b, d, l]
        lr_chunk, log_decay_chunk, log_beta_chunk,  # [b, l, 1]
        w0, w1, w2,  # w0 & w2: [b, dh, d]; w1: [b, d, dh]
        m_init_w0, m_init_w1, m_init_w2,  # prev momentum states m_prev for each param
        w0_norm, w1_norm, w2_norm
    ):
        # For RMSNorm
        ttt_norm_weight = self.ttt_norm_weight.unsqueeze(-1)
        ttt_norm_bias = self.ttt_norm_bias.unsqueeze(-1)

        # Retrieve memory (Read out)
        # layernorm + residual will be done once after all chunks are processed
        h = torch.bmm(w2, q_chunk)
        gate = F.silu(torch.bmm(w0, q_chunk))
        out = torch.bmm(w1, gate * h)

        # Forward
        # MLPs
        Z0 = torch.bmm(w0, k_chunk)  # [b, dh, l]
        Z2 = torch.bmm(w2, k_chunk)  # [b, dh, l]
        G = F.silu(Z0)  # [b, dh, l]
        H = G * Z2  # [b, dh, l]
        pre = torch.bmm(w1, H).float() # we use residual here  [b, d, l]
        # RMSNorm + Residual (Residual in already included in the v_chunk, see self.ln_restruction_target)
        O, mu, inv_s = layernorm_fwd(pre, ttt_norm_weight, ttt_norm_bias)

        # MSE loss
        Err = O - v_chunk  # [b, d, l]

        # Closed-form time weights
        log_alpha, sbeta, carry_u, carry_m = build_token_weights_chunk_log(log_beta_chunk, log_decay_chunk)
        carry_u, carry_m = carry_u.to(k_chunk.dtype), carry_m.to(v_chunk.dtype)
        alpha = torch.exp(log_alpha)  #
        alpha_lr = (alpha * lr_chunk).to(k_chunk.dtype)
        Err_alpha = Err * alpha_lr.transpose(1, 2)

        # LayerNorm backward
        grad_pre = layernorm_bwd(pre, Err_alpha, ttt_norm_weight, mu, inv_s)

        # Backprop to hidden terms
        B = torch.bmm(w1.transpose(1, 2), grad_pre)  # [b, dh, l]
        sig = torch.sigmoid(Z0)
        silu_prime = sig * (1.0 + Z0 * (1.0 - sig))  # [b, dh, l]
        dZ2 = B * G  # [b, dh, l]
        dZ0 = B * Z2 * silu_prime  # [b, dh, l]

        # Calculating dw1
        dw1 = -torch.bmm(grad_pre, H.transpose(1, 2))

        # Calculating dw0 & dw2
        k_t = k_chunk.transpose(1, 2)  # [b, l, d]
        dw2 = -torch.bmm(dZ2, k_t)
        dw0 = -torch.bmm(dZ0, k_t)

        if self.verbose:
            rel_step = dw0.norm() / w0.norm()
            print(f"before clipping rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
            print(f"carry_u: {carry_u[:, 0, 0].tolist()}")
            print(f"carry_m: {carry_m[:, 0, 0].tolist()}")
            print(f"dw0 head norm before clip {dw0.norm(dim=(1,2))}")

        # gradient clip
        dw0 = gradient_clip(dw0)
        dw1 = gradient_clip(dw1)
        dw2 = gradient_clip(dw2)

        momentum_w0 = carry_m * m_init_w0
        momentum_w1 = carry_m.transpose(1, 2) * m_init_w1
        momentum_w2 = carry_m * m_init_w2

        # We now calculate the weights for the last token in the chunk
        upd_w0 = carry_u * w0 + momentum_w0 + dw0
        upd_w1 = carry_u.transpose(1, 2) * w1 + momentum_w1 + dw1
        upd_w2 = carry_u * w2 + momentum_w2 + dw2

        # Weight normalization (same as in test training done right)
        upd_w0 = upd_w0 / (upd_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        upd_w1 = upd_w1 / (upd_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        upd_w2 = upd_w2 / (upd_w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        if self.verbose:
            rel_step = dw0.norm() / w0.norm()
            print(f"rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
            print(f"dw0 head norm after clip {dw0.norm(dim=(1,2))}")
            print(f"MSE: {Err.pow(2).mean()}")

        # We now calculate the momentum of the last token for the next chunk
        prod_beta = log_beta_chunk.sum(dim=1, keepdim=True).exp().to(k_chunk.dtype)  # [b,1,1]
        beta_lr = (sbeta * lr_chunk).to(k_chunk.dtype)

        # We might want to prune tokens which do not contribute to the momentum update
        n_keep = pick_tail_len_pow2(beta_lr)
        if n_keep > 0:
            H = H[..., -n_keep:]
            k_t = k_t[:, -n_keep:]
            dZ0 = dZ0[..., -n_keep:]
            dZ2 = dZ2[..., -n_keep:]
            sbeta = sbeta[:, -n_keep:]
            alpha = alpha[:, -n_keep:]
            grad_pre = grad_pre[..., -n_keep:]

            ratio = (sbeta / (alpha + 1e-12)).transpose(1, 2).to(k_chunk.dtype)  # [b, 1, l]
            grad_pre_beta = grad_pre * ratio  # [b, d, l]
            dZ2_beta = dZ2 * ratio
            dZ0_beta = dZ0 * ratio

            m_w1_add = -torch.bmm(grad_pre_beta, H.transpose(1, 2))
            m_w0_add = -torch.bmm(dZ0_beta, k_t)  # [b, dh, d]
            m_w2_add = -torch.bmm(dZ2_beta, k_t)  # [b, dh, d]

            if self.verbose:
                rel_step = m_w0_add.norm() / m_init_w0.norm()
                print(f"before clipping rel_step: {rel_step}, m add w0 norm {m_w0_add.norm()}, m init w0 norm {m_init_w0.norm()}")

            m_w0_add = gradient_clip(m_w0_add, max_norm=1.0)
            m_w1_add = gradient_clip(m_w1_add, max_norm=1.0)
            m_w2_add = gradient_clip(m_w2_add, max_norm=1.0)

            if self.verbose:
                rel_step = m_w0_add.norm() / m_init_w0.norm()
                print(f"rel_step: {rel_step}, m add w0 norm {m_w0_add.norm()}, m init w0 norm {m_init_w0.norm()}")

            m_next_w1 = prod_beta.transpose(1, 2) * m_init_w1 + m_w1_add  # [b, d, dh]
            m_next_w0 = prod_beta * m_init_w0 + m_w0_add  # [b, dh, d]
            m_next_w2 = prod_beta * m_init_w2 + m_w2_add  # [b, dh, d]
        else:
            m_next_w1 = prod_beta * m_init_w1  # [b, d, dh]
            m_next_w0 = prod_beta * m_init_w0  # [b, dh, d]
            m_next_w2 = prod_beta * m_init_w2  # [b, dh, d]

        return out, upd_w0, upd_w1, upd_w2, m_next_w0, m_next_w1, m_next_w2

    def update_and_retrieve_memory(self, q, k, v, lr, log_decay, log_momentum, recurrent_state):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        bh, d, orig_seq_len = q.shape

        if recurrent_state is not None:
            w0, w1, w2, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1, m_next_w2 = recurrent_state

            # we have to zero pad q as well, otherwise the shape mismatch
            pad_q = torch.zeros_like(buf_k)
            q = torch.cat([pad_q, q], dim=-1)
            k = torch.cat([buf_k, k], dim=-1)
            v = torch.cat([buf_v, v], dim=-1)
            lr = torch.cat([buf_lr, lr], dim=1)
            log_decay = torch.cat([buf_log_decay, log_decay], dim=1) if log_decay is not None else None
            log_momentum = torch.cat([buf_log_momentum, log_momentum], dim=1) if log_momentum is not None else None
        else:
            w0, w1, w2 = self.w0, self.w1, self.w2
            m_next_w0 = torch.zeros_like(w0)
            m_next_w1 = torch.zeros_like(w1)
            m_next_w2 = torch.zeros_like(w2)


        seq_len = k.shape[-1]
        rem = seq_len % self.chunk_size

        q_chunks, k_chunks, v_chunks = [
            x.split(self.chunk_size, dim=-1) for x in [q, k, v]
        ]

        lr_chunks, log_decay_chunks, log_momentum_chunks = ([
            x.split(self.chunk_size, dim=1) for x in [lr, log_decay, log_momentum]
        ])

        num_chunks = len(k_chunks)
        w0_norm = w0.norm(dim=2, keepdim=True)
        w1_norm = w1.norm(dim=2, keepdim=True)
        w2_norm = w2.norm(dim=2, keepdim=True)

        # Read out
        output = torch.zeros_like(q)
        s_index = 0

        # Choose correct update chunk function
        if self.ttt_momentum == "none":
            update_chunk_fn = self.update_chunk_wo_momentum
        else:
            update_chunk_fn = self.update_chunk

        for i in range(num_chunks):
            e_index = s_index + self.chunk_size
            # We do not update the last chunk if its length is smaller than chunk_size
            if i == num_chunks - 1 and rem != 0:
                continue

            if self.verbose: print(f"{i+1}-th chunk")
            out, w0, w1, w2, m_next_w0, m_next_w1, m_next_w2 = update_chunk_fn(
                q_chunks[i], k_chunks[i], v_chunks[i],
                lr_chunks[i], log_decay_chunks[i], log_momentum_chunks[i],
                w0, w1, w2,
                m_next_w0, m_next_w1, m_next_w2,
                w0_norm, w1_norm, w2_norm
            )

            output[..., s_index:e_index] = out
            s_index = e_index

        if rem != 0:
            # processing last chunk, direct read out
            q_chunk = q_chunks[-1]
            h = torch.bmm(w2, q_chunk)
            gate = F.silu(torch.bmm(w0, q_chunk))
            tail = torch.bmm(w1, gate * h)
            output[:, :, s_index:] = tail

        # Do the layernorm + residual once
        ttt_norm_weight = self.ttt_norm_weight.unsqueeze(1)
        ttt_norm_bias = self.ttt_norm_bias.unsqueeze(1)
        x = output.transpose(1, 2)  # b l d
        x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-8)
        out = x_norm * ttt_norm_weight.float() + ttt_norm_bias.float()
        out = out.to(q.dtype) + q.transpose(1, 2)
        out = out[:, -orig_seq_len:, :]  # exclude pads if necessary

        # Prepare buf
        if rem != 0:
            buf_k = k_chunks[-1]
            buf_v = v_chunks[-1]
            buf_lr = lr_chunks[-1]
            buf_log_decay = log_decay_chunks[-1]
            buf_log_momentum = log_momentum_chunks[-1]
        else:
            buf_k = q.new_zeros(bh, d, 0)
            buf_v = q.new_zeros(bh, d, 0)
            buf_lr = q.new_zeros(bh, 0, lr.shape[-1])
            buf_log_decay = q.new_zeros(bh, 0, log_decay.shape[-1]) if log_decay is not None else None
            buf_log_momentum = q.new_zeros(bh, 0, log_momentum.shape[-1])

        return out, (w0, w1, w2, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1, m_next_w2)

    def update_and_retrieve_memory_generate(self, q, k, v, lr, log_decay, log_momentum, recurrent_state):
        assert recurrent_state is not None
        bh, l, d = q.shape
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # b d l

        # Read from cache
        w0, w1, w2, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1, m_next_w2 = recurrent_state

        ttt_norm_weight = self.ttt_norm_weight.unsqueeze(1)
        ttt_norm_bias = self.ttt_norm_bias.unsqueeze(1)

        # In current version, we do not update fast weights and the cache for generation (do not want to remember own answers)
        h = torch.bmm(w2, q)
        gate = F.silu(torch.bmm(w0, q))
        out = torch.bmm(w1, gate * h)  # [b d l]
        x = out.transpose(1, 2)  # [b l d]
        x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-8)
        out = x_norm * ttt_norm_weight.float() + ttt_norm_bias.float()
        out = out.to(q.dtype) + q.transpose(1, 2)
        return out, recurrent_state
