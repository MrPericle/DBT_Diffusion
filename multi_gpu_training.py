"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
import torchio as tio
import os
import csv
import itertools
import time
import h5py

import numpy as np
from pydicom import dcmread
from torch.utils import data
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from text import tokenize, bert_embed, BERT_MODEL_DIM

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed

# helpers functions

PREPROCESSING_TRANSORMS=tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.transforms.Resize((256,256,32)),
    tio.transforms.Resize((128,128,16)),
    tio.transforms.Resize((64,64,8))
])

TRAIN_TRANSFORMS=tio.Compose([
    # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
    # 0, 0, 3), translation=(4, 4, 0)),
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

def exists(x):
    return x is not None

def loss_logger(log):
    with open("../DDP_results/log.csv" , "a+",newline='') as log_loss:
        writer=csv.writer(log_loss)
        writer.writerows(log)

def is_odd(n):
    return (n % 2)==1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Embedding layer per memorizzare i bias di attenzione relativi
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Calcola i valori del bucket in base alla posizione relativa
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        #print("Relative position bias ok?")
        # Genera indici posizionali per query e chiavi
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)

        # Calcola posizioni relative tra query e chiavi
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')

        # Calcola il bucket della posizione relativa per ogni coppia di posizioni
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)

        # Recupera i bias di attenzione relativi dallo strato di embedding
        values = self.relative_attention_bias(rp_bucket)

        # Riorganizza il tensore per la compatibilità con il meccanismo di attenzione
        #print("Relative position bias ok")
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        #print("Residial ok")
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        #print("Sinusoidal positional embedding ok?")
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        #print("Sinusoidal positional embedding ok")
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        #print("Layer norm ok?")
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        #print("Layer norm ok")
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        #print("PreNorm ok?")
        x = self.norm(x)
        #print("PreNorm ok")
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        #print("Block ok?")
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        #print("Block ok")
        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        #print("ResNet Block ok?")
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        #print("ResNet Block ok")
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        #print("SpatialLinearAttention ok?")
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        #print("SpatialLinearAttention ok")
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        #print("EinopsToAndFrom ok?")
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        #print("EinopsToAndFrom ok")
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        #print("Attention ok?")
        n, device = x.shape[-2], x.device
        #print("Attention 0")
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #print("Attention 1")
        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            #print("Attention ok")
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
        # scale
        #print("Attention 2")
        q = q * self.scale
        # rotate positions into queries and keys for time attention
        #print("Attention 3")
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias
        #print("Attention 4")
        if exists(pos_bias):
            sim = sim + pos_bias
        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        # numerical stability
        #print("Attention 5")
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        #print("Attention 6")
        attn = sim.softmax(dim = -1)
        #print("Attention 7")
        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        #print("Attention 8")
        out = rearrange(out, '... h n d -> ... n (h d)')
        #print("Attention ok")
        return self.to_out(out)

# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        attn_heads = 8,
        attn_dim_head = 32,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        block_type = 'resnet',
        resnet_groups = 8
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb)) #TU SEI IL MOTIVO DEL CRASH

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many slices of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        #print("Unet ok?")
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        #print(f"initial shape: {x.shape}")
        x = self.init_conv(x)
        #print(f"Unet step 1 -> {x.shape}")
        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias) #TU SEI IL PROBLEMA
        #print(f"Unet step 2 ->{x.shape}")
        r = x.clone()
        #print(f"Unet step 3 ->{x.shape}")
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        #print(f"Unet step 4 ->{x.shape}")
        # classifier free guidance
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device = device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim = -1)

        h = []
        #print(f"Unet step 5 -> {x.shape}")
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            #print(f"Unet step 5.a -> {x.shape}")
            x = block2(x, t)
            #print(f"Unet step 5.b -> {x.shape}")
            x = spatial_attn(x)
            #print(f"Unet step 5.c -> {x.shape}")
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            #print(f"Unet step 5.d -> {x.shape}")
            h.append(x)
            x = downsample(x)
            #print(f"Unet step 5.e -> {x.shape}")
        #print(f"Unet step 6 -> {x.shape}")
        x = self.mid_block1(x, t)
        #print(f"Unet step 7 -> {x.shape}")
        x = self.mid_spatial_attn(x)
        #print(f"Unet step 8 -> {x.shape}")
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        #print(f"Unet step 9 -> {x.shape}")
        x = self.mid_block2(x, t)
        #print(f"Unet step 10 -> {x.shape}")
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            #print(f"Unet step 10.a -> {x.shape}")
            x = block1(x, t)
            #print(f"Unet step 10.b -> imag shape = {x.shape}")
            x = block2(x, t)
            #print(f"Unet step 10.c -> {x.shape}")
            x = spatial_attn(x)
            #print(f"Unet step 10.d -> {x.shape}")
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            #print(f"Unet step 10.e -> {x.shape}")
            x = upsample(x)
            #print(f"Unet step 10.f -> {x.shape}")

        x = torch.cat((x, r), dim = 1)
        #print(f"Unet step 11 -> {x.shape}")
        #print("Unet ok")
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_slices,
        text_use_bert_cls = False,
        channels = 1,
        timesteps = 1000,
        loss_type = 'l2',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9,
        gpu_id
    ):
        super().__init__()
        self.channels = channels
        self.gpu_id = gpu_id
        self.image_size = image_size
        self.num_slices = num_slices
        self.denoise_fn = DDP(denoise_fn,device_ids=[self.gpu_id],output_device = self.gpu_id).cuda()

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32).cuda())

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.module.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond = None, cond_scale = 1., batch_size = 16):
        device = next(self.denoise_fn.parameters()).device #forse mettere cuda?

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_slices = self.num_slices
        return self.p_sample_loop((batch_size, channels, num_slices, image_size, image_size), cond = cond, cond_scale = cond_scale)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr = self.text_use_bert_cls)
            cond = cond.to(device)
        start_time = time.time()
        x_recon = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)
        print(f"time for a unet forward : {time.time()-start_time}")
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        #print("GaussianDiffusion ok?")
        b, device, img_size, = x.shape[0], x.device, self.image_size
        print(device)
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_slices, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        #print("GaussianDiffusion ok")
        return self.p_losses(x, t, *args, **kwargs)


# tensor of shape (channels, slice, height, width) -> tensor

def tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

def save_4d_tensor_as_hdf5(tensor, output_file):
    # Assuming tensor shape is (channel, depth, height, width)
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('tensor_data', data=tensor)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_slices(t, *, slices):
    f = t.shape[1]

    if f == slices:
        return t

    if f > slices:
        return t[:, :slices]

    return F.pad(t, (0, 0, 0, 0, 0, slices - f))

class Duke_DBT_Dataset(Dataset):
    def __init__(self,root_dir,train = True,Train_Transform = True):
        self.root_dir = root_dir
        self.train = train
        #self.img_path = os.path.join(root_dir,'train') if self.train == True else os.path.join(root_dir,'test') #aggiungere opzione per validation ma al momento manca nel dataset
        self.img_path = [f"{self.root_dir}/{filename}" for filename in os.listdir(self.root_dir) if filename.endswith('.h5')]
        #self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS if Train_Transform else None

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        with h5py.File(self.img_path[index], 'r') as hf:
            img = hf['tensor_data'][:]

        img=torch.from_numpy(img)
        img = torch.squeeze(img,dim=0)
        return img


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        num_slices = 8,
        train_batch_size = 8,
        train_lr = 1e-4,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = '../DDP_results',
        num_sample_rows = 4,
        max_grad_norm = None,
        optimizer,
        gpu_id,
        num_epoch = 6
    ):
        super().__init__()
        self.gpu_id = gpu_id

        self.model = diffusion_model
        self.ema = EMA(ema_decay)


        self.ema_model = copy.deepcopy(self.model)


        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
       # self.train_num_steps = train_num_steps
        self.num_epoch = num_epoch

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_slices = diffusion_model.num_slices

        self.ds = Duke_DBT_Dataset(folder)

        if self.gpu_id==0:
            print(f'found {len(self.ds)} images as DICOM files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 .dcm to start training (although 1 is not great, try 100k)'

        self.dl = data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=False, sampler=DistributedSampler(self.ds),pin_memory = True)
        self.opt = optimizer

        self.step = 0
        self.epoch = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()
        self.model = DDP(self.model,device_ids=[self.gpu_id]).cuda()
        self.ema_model = DDP(self.ema_model,device_ids=[self.gpu_id]).cuda()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


    def save(self, milestone):
        data = {
            'epoch': self.epoch,
            'step': self.step,
            'model': self.model.module.state_dict(),
            'ema': self.ema_model.module.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))


    def load(self, milestone, **kwargs):
        print("LOADING FUNCTION")
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)
        torch.distributed.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.gpu_id}
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),map_location=map_location)

        self.step = data['step']
        self.epoch = data['epoch']

        self.model.module.load_state_dict(data['model'], **kwargs)

        self.ema_model.module.load_state_dict(data['ema'], **kwargs)

        self.scaler.load_state_dict(data['scaler'])
        return milestone


    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = loss_logger
    ):
        assert callable(log_fn)
        if self.gpu_id==0:
            print("Start training\n")
        loss_logger = []

        start_step = self.step
        self.model.train()
        for epoch in range(self.epoch,self.num_epoch):
            self.dl.sampler.set_epoch(epoch)
            if self.gpu_id == 0:
                pbar = tqdm(total=len(self.dl), desc='Training Progress')
                pbar.set_description(f'epoch:{self.epoch}')
            for step,data in enumerate(self.dl):
                data = data.to(self.gpu_id)
                step_time = time.time()
                for i in range(self.gradient_accumulate_every):
                    with autocast(enabled = self.amp):
                        loss = self.model(
                            data,
                            prob_focus_present = prob_focus_present,
                            focus_present_mask = focus_present_mask
                        )
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                if self.gpu_id == 0:
                    print(f'{self.step}: {loss.item()}')
                    pbar.update(1)


                if exists(self.max_grad_norm):
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                if self.step % self.update_ema_every == 0 and self.gpu_id==0:
                    self.step_ema()

                self.step +=1
                print(f"total time for a step : {time.time()-step_time}")

            self.epoch +=1
            self.step = 0
            if self.gpu_id == 0 :
                self.save(self.epoch)
                loss_logger.append((self.epoch,loss.item()))
                log_fn(loss_logger)
                loss_logger.clear()

            if self.gpu_id == 0:
                pbar.close()
        print(f'training completed')
