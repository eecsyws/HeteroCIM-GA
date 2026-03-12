"""
量化模块
实现FakeQuantize和量化层，支持per-layer配置和NVM噪声注入
基于权威的INT8二进制补码位平面噪声注入方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from timm.models.vision_transformer import Attention as TimmAttention

from config import HardwareConfig


# ============================================================
# 对称按通道伪量化 / 反量化（支持任意位宽INT3-INT8）
# ============================================================
def per_channel_fake_quant_int8(weight, ch_axis=0, num_bits=8):
    """
    对称 per-channel fake quant
    支持任意位宽（INT3-INT8）
    zero_point 固定为 0
    """
    # INTn范围: [-(2^(n-1)), 2^(n-1)-1]
    # 例如: INT8: [-128, 127], INT5: [-16, 15], INT4: [-8, 7]
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    permute_dims = [ch_axis] + [i for i in range(weight.dim()) if i != ch_axis]
    inv_permute = [permute_dims.index(i) for i in range(len(permute_dims))]

    w = weight.permute(permute_dims)
    c = w.size(0)
    w_flat = w.reshape(c, -1)

    max_abs = w_flat.abs().max(dim=1)[0]
    scale = max_abs / float(qmax)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    view_shape = [c] + [1] * (weight.dim() - 1)
    scale_b = scale.view(view_shape).to(weight.device)

    q = torch.round(w / scale_b).clamp(qmin, qmax)
    return q.permute(inv_permute).contiguous(), scale.to(weight.device)


def per_channel_dequant_int8(q, scale, ch_axis=0):
    """反量化"""
    permute_dims = [ch_axis] + [i for i in range(q.dim()) if i != ch_axis]
    inv_permute = [permute_dims.index(i) for i in range(len(permute_dims))]

    q_p = q.permute(permute_dims)
    c = q_p.size(0)

    view_shape = [c] + [1] * (q.dim() - 1)
    scale_b = scale.view(view_shape).to(q.device)

    w_deq = q_p * scale_b
    return w_deq.permute(inv_permute).contiguous()


# ============================================================
# 二进制补码位平面高斯噪声（支持任意位宽INT3-INT8）
# ============================================================
def add_bitwise_gaussian_noise(q_float, sigma, num_bits, noise_bits):
    """
    针对 n 位二进制补码做位平面噪声（已修正版）
    逻辑：
        - 低位 (0 到 noise_bits-1) 为 NVM，注入噪声
        - 高位 (noise_bits 到 num_bits-1) 为 SRAM，无噪声
    """
    if sigma <= 0 or noise_bits <= 0:
        return q_float

    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    q_int = torch.round(q_float).clamp(qmin, qmax).to(torch.int32)

    # 转为补码对应的无符号表示
    offset = 2 ** num_bits
    q_twos = torch.where(q_int < 0, q_int + offset, q_int).to(torch.int32)

    q_noised = torch.zeros_like(q_float, dtype=torch.float32)

    # 遍历所有有效 bit 位
    for bit in range(num_bits):
        mask = 1 << bit
        bit_plane = ((q_twos & mask) >> bit).float()

        # 计算位权
        if bit == num_bits - 1:
            bit_weight = -float(2 ** (num_bits - 1)) # 符号位权重
        else:
            bit_weight = float(1 << bit)             # 数值位权重

        # --- 修正后的逻辑 ---
        # 如果当前位属于低位的 noise_bits 范围，则判定为 NVM，注入噪声
        if bit < noise_bits:
            # NVM bits (LSB 部分)：注入噪声
            noise = torch.randn_like(q_noised) * sigma
            factor = 1.0 + noise.clamp(-3 * sigma, 3 * sigma)
            q_noised += bit_plane * factor * bit_weight
        else:
            # SRAM bits (MSB 部分)：保持纯净
            q_noised += bit_plane * bit_weight

    return q_noised.clamp(qmin, qmax)


def quant_noise_dequant_weight(weight, sigma, ch_axis=0, num_bits=8, noise_bits=8):
    """
    完整的量化-噪声-反量化流程（支持任意位宽INT3-INT8）

    参数:
        weight: 权重张量
        sigma: 噪声标准差
        ch_axis: channel维度
        num_bits: 量化总位数
        noise_bits: NVM噪声位数
    """
    q, scale = per_channel_fake_quant_int8(weight, ch_axis=ch_axis, num_bits=num_bits)
    q_noised = add_bitwise_gaussian_noise(q, sigma=sigma, num_bits=num_bits, noise_bits=noise_bits)
    return per_channel_dequant_int8(q_noised, scale, ch_axis=ch_axis)


# ============================================================
# 通用量化函数（支持其他位宽）
# ============================================================
def fake_quant_tensor(
    x: torch.Tensor,
    signed: bool,
    num_bits: int,
    per_channel: bool = False,
    ch_axis: int = -1,
    eps: float = 1e-8,
):
    """
    对张量进行伪量化（动态量化）
    用于激活量化或非INT8的权重量化
    """
    if num_bits <= 0 or num_bits > 16:
        return x

    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)

    if per_channel:
        if ch_axis < 0:
            ch_axis = x_fp32.dim() + ch_axis
        feat_dim = [d for d in range(x_fp32.dim()) if d != ch_axis]
    else:
        feat_dim = None

    if signed:
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1
        if per_channel:
            max_abs = x_fp32.abs().amax(dim=feat_dim, keepdim=True)
        else:
            max_abs = x_fp32.abs().max()
        max_abs = torch.clamp(max_abs, min=eps)
        scale = max_abs / float(qmax)
        zero_point = torch.zeros_like(scale) if per_channel else 0.0
    else:
        qmin = 0
        qmax = (2 ** num_bits) - 1
        if per_channel:
            rmin = x_fp32.amin(dim=feat_dim, keepdim=True)
            rmax = x_fp32.amax(dim=feat_dim, keepdim=True)
        else:
            rmin = x_fp32.min()
            rmax = x_fp32.max()
        scale = (rmax - rmin) / float(qmax - qmin)
        scale = torch.clamp(scale, min=eps)
        zero_point = torch.round(qmin - rmin / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)

    q = torch.round(x_fp32 / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    x_q = (q - zero_point) * scale
    return x_q.to(orig_dtype)


def fake_quant_tensor_per_token(
    x: torch.Tensor,
    signed: bool,
    num_bits: int,
    is_conv2d: bool = False,
    eps: float = 1e-8,
):
    """
    Per-token量化（用于激活）
    """
    if num_bits <= 0 or num_bits > 16:
        return x

    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)

    if is_conv2d:
        reduce_dims = (2, 3)
    else:
        reduce_dims = (-1,)

    if signed:
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1
        max_abs = x_fp32.abs().amax(dim=reduce_dims, keepdim=True)
        max_abs = torch.clamp(max_abs, min=eps)
        scale = max_abs / float(qmax)
        zero_point = torch.zeros_like(scale)
    else:
        qmin = 0
        qmax = (2 ** num_bits) - 1
        rmin = x_fp32.amin(dim=reduce_dims, keepdim=True)
        rmax = x_fp32.amax(dim=reduce_dims, keepdim=True)
        scale = (rmax - rmin) / float(qmax - qmin)
        scale = torch.clamp(scale, min=eps)
        zero_point = torch.round(qmin - rmin / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)

    q = torch.round(x_fp32 / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    x_q = (q - zero_point) * scale
    return x_q.to(orig_dtype)


# ============================================================
# 权重量化+噪声注入（统一接口）
# ============================================================
def fake_quant_weight_with_noise(
    w: torch.Tensor,
    signed: bool,
    num_bits: int,
    per_channel: bool,
    ch_axis: int,
    noise_bits: int,
    sigma: float,
):
    """
    对权重进行量化并注入噪声

    参数:
        w: 权重张量
        signed: 是否有符号
        num_bits: 量化位宽
        per_channel: 是否per-channel
        ch_axis: channel维度
        noise_bits: NVM噪声位数（低bit位注入噪声）
        sigma: 噪声标准差

    逻辑:
        - SRAM bits = num_bits - noise_bits（高bit位，无噪声）
        - NVM bits = noise_bits（低bit位，有噪声）
        - 例如INT5(3,2): SRAM=2(高2位), NVM=2(低2位)
    """
    if num_bits <= 0 or num_bits > 16:
        return w

    # 确保noise_bits不超过num_bits
    noise_bits = min(noise_bits, num_bits)

    orig_dtype = w.dtype
    w_fp32 = w.to(torch.float32)

    # 如果noise_bits > 0且sigma > 0，注入噪声
    if noise_bits > 0 and sigma > 0 and signed and per_channel:
        w_q = quant_noise_dequant_weight(
            w_fp32,
            sigma=sigma,
            ch_axis=ch_axis,
            num_bits=num_bits,
            noise_bits=noise_bits
        )
        return w_q.to(orig_dtype)

    # noise_bits == 0 或 sigma == 0：仅量化，无噪声
    else:
        return fake_quant_tensor(
            w_fp32,
            signed=signed,
            num_bits=num_bits,
            per_channel=per_channel,
            ch_axis=ch_axis,
        ).to(orig_dtype)


# ============================================================
# FakeQuant Wrapper（Linear / Conv2d）
# ============================================================
class FakeQuantWrapper(nn.Module):
    """
    量化Wrapper，包装Linear或Conv2d层
    支持权重量化、激活量化和NVM噪声注入
    """
    def __init__(
        self,
        module: nn.Module,
        quant_w: bool = True,
        quant_a: bool = True,
        w_signed: bool = True,
        a_signed: bool = True,
        w_per_channel: bool = True,
        a_per_channel: bool = False,
        num_bits: int = 8,
        noise_bits: int = 0,
        use_static_activation: bool = False,
        full_name: str = "",
    ):
        super().__init__()
        self.module = module
        self.quant_w = quant_w
        self.quant_a = quant_a
        self.w_signed = w_signed
        self.a_signed = a_signed
        self.w_per_channel = w_per_channel
        self.a_per_channel = a_per_channel
        self.num_bits = num_bits
        self.noise_bits = noise_bits
        self.full_name = full_name

        # 激活静态量化相关
        self.use_static_activation = use_static_activation
        self.collect_activation_stats = False
        self.a_min = None
        self.a_max = None
        self.a_scale = None
        self.a_zero_point = None

    def forward(self, x):
        # 激活量化
        if self.quant_a:
            if not self.collect_activation_stats:
                x = fake_quant_tensor_per_token(
                    x,
                    signed=self.a_signed,
                    num_bits=self.num_bits,
                    is_conv2d=isinstance(self.module, nn.Conv2d),
                )

        # 权重量化 + 噪声
        w = getattr(self.module, "weight", None)
        if self.quant_w and w is not None:
            if isinstance(self.module, (nn.Conv2d, nn.Linear)):
                w_q = fake_quant_weight_with_noise(
                    w,
                    signed=self.w_signed,
                    num_bits=self.num_bits,
                    per_channel=self.w_per_channel,
                    ch_axis=0,
                    noise_bits=self.noise_bits,
                    sigma=HardwareConfig.NOISE_SIGMA,
                )
            else:
                w_q = fake_quant_tensor(
                    w,
                    signed=self.w_signed,
                    num_bits=self.num_bits,
                    per_channel=False,
                )
        else:
            w_q = w

        # 执行计算
        if isinstance(self.module, nn.Linear):
            return F.linear(x, w_q, self.module.bias)
        elif isinstance(self.module, nn.Conv2d):
            return F.conv2d(
                x,
                w_q,
                self.module.bias,
                stride=self.module.stride,
                padding=self.module.padding,
                dilation=self.module.dilation,
                groups=self.module.groups,
            )
        else:
            return self.module(x)


# ============================================================
# Attention Wrapper：支持 q/k/v、qk、av 的 per-subop bits
# ============================================================
class FakeQuantAttention(nn.Module):
    """
    量化Attention模块
    支持Q/K/V、QK^T、AV的独立量化配置
    """
    def __init__(
        self,
        attn: TimmAttention,
        full_name: str,
        w_signed: bool,
        a_signed: bool,
        w_per_channel: bool,
        layer_config: Dict[str, Dict],  # {'.q': {'quant_bits': 8, 'nvm_bits': 8}, ...}
        attn_qkv_quant: bool = True,
        attn_matmul_quant: bool = True,
    ):
        super().__init__()
        self.attn = attn
        self.full_name = full_name
        self.w_signed = w_signed
        self.a_signed = a_signed
        self.w_per_channel = w_per_channel
        self.layer_config = layer_config
        self.attn_qkv_quant = attn_qkv_quant
        self.attn_matmul_quant = attn_matmul_quant

    def _get_config_for_subop(self, sub_tag: str):
        """获取子操作的配置"""
        return self.layer_config.get(sub_tag, {'quant_bits': 8, 'nvm_bits': 0})

    def _quant_linear_branch(self, x, w, b, key: str):
        """量化Q/K/V分支"""
        if not self.attn_qkv_quant:
            return F.linear(x, w, b)

        sub_tag = f".{key}"
        config = self._get_config_for_subop(sub_tag)
        bits = config['quant_bits']
        noise_bits = config['nvm_bits']

        # 激活量化
        x_q = fake_quant_tensor_per_token(
            x,
            signed=self.a_signed,
            num_bits=bits,
            is_conv2d=False,
        )

        # 权重量化 + 噪声
        w_q = fake_quant_weight_with_noise(
            w,
            signed=self.w_signed,
            num_bits=bits,
            per_channel=self.w_per_channel,
            ch_axis=0,
            noise_bits=noise_bits,
            sigma=HardwareConfig.NOISE_SIGMA,
        )

        return F.linear(x_q, w_q, b)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape

        # 拆分qkv权重
        qkv_linear = self.attn.qkv
        W = qkv_linear.weight  # [3*C, C]
        b = qkv_linear.bias    # [3*C]

        W_q, W_k, W_v = torch.chunk(W, 3, dim=0)
        if b is not None:
            b_q, b_k, b_v = torch.chunk(b, 3, dim=0)
        else:
            b_q = b_k = b_v = None

        # XW_Q / XW_K / XW_V
        q = self._quant_linear_branch(x, W_q, b_q, key="q")
        k = self._quant_linear_branch(x, W_k, b_k, key="k")
        v = self._quant_linear_branch(x, W_v, b_v, key="v")

        # reshape到 (B, heads, N, head_dim)
        H = self.attn.num_heads
        head_dim = C // H
        q = q.reshape(B, N, H, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, H, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, H, head_dim).permute(0, 2, 1, 3)

        q = q * self.attn.scale

        # QK^T
        if self.attn_matmul_quant:
            config_qk = self._get_config_for_subop(".qk")
            bits_qk = config_qk['quant_bits']
            q_q = fake_quant_tensor(
                q,
                signed=self.a_signed,
                num_bits=bits_qk,
                per_channel=False,
            )
            k_q = fake_quant_tensor(
                k,
                signed=self.a_signed,
                num_bits=bits_qk,
                per_channel=False,
            )
            attn_scores = q_q @ k_q.transpose(-2, -1)
        else:
            attn_scores = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn = attn_scores.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)

        # A·V
        if self.attn_matmul_quant:
            config_av = self._get_config_for_subop(".av")
            bits_av = config_av['quant_bits']
            attn_q = fake_quant_tensor(
                attn,
                signed=False,
                num_bits=bits_av,
                per_channel=False,
            )
            v_q = fake_quant_tensor(
                v,
                signed=self.a_signed,
                num_bits=bits_av,
                per_channel=False,
            )
            out = attn_q @ v_q
        else:
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)

        # Output projection with quantization
        config_proj = self._get_config_for_subop(".proj")
        bits_proj = config_proj['quant_bits']
        noise_bits_proj = config_proj['nvm_bits']

        if bits_proj > 0 and bits_proj <= 16:
            # 激活量化
            out_q = fake_quant_tensor_per_token(
                out,
                signed=self.a_signed,
                num_bits=bits_proj,
                is_conv2d=False,
            )
            # 权重量化 + 噪声
            proj_w = self.attn.proj.weight
            proj_b = self.attn.proj.bias
            proj_w_q = fake_quant_weight_with_noise(
                proj_w,
                signed=self.w_signed,
                num_bits=bits_proj,
                per_channel=self.w_per_channel,
                ch_axis=0,
                noise_bits=noise_bits_proj,
                sigma=HardwareConfig.NOISE_SIGMA,
            )
            out = F.linear(out_q, proj_w_q, proj_b)
        else:
            out = self.attn.proj(out)

        out = self.attn.proj_drop(out)
        return out
