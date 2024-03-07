"""
DBT_Diffusion: Denoising Diffusion Probabilistic Models for Digital Breast Tomosynthesis Augmentation

Author: MrPericle
Email: l.pergamo001@studenti.uniparthenope.it
Version: 1.0.0

Description:
This script implements multi-GPU training for the DBT_Diffusion project. It utilizes Denoising Diffusion Probabilistic Models to generate synthetic DBT samples for dataset augmentation.

Usage:
torchrun --standalone --nnodes=<N_NODES> --nproc_per_node=<YOUR_GPUS> multi_gpu_training.py
"""

import torch
from DBT_multigpu_diffusion import Unet3D, GaussianDiffusion, Trainer,Duke_DBT_Dataset
import os
import time
import copy
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank):
    init_process_group(backend="nccl")
    torch.cuda.set_device(rank)


def main():
    local_rank=int(os.environ['LOCAL_RANK'])
    ddp_setup(rank=local_rank)
    if local_rank==0 and torch.cuda.is_available():
        print(f"CUDA is available. Using {torch.cuda.device_count()} GPU.")
    elif local_rank==0:
        print("CUDA is not available. Using CPU.")

    model=Unet3D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
    )

    diffusion=GaussianDiffusion(
        model,
        image_size=64,
        num_slices=8,
        timesteps=250,   # number of steps
        loss_type='l1',    # L1 or L2
        gpu_id=local_rank
    )

    train_dataset=Duke_DBT_Dataset("/projects/data/medical_imaging/duke2/BCS_DBT_Train")
    optimizer=Adam(diffusion.parameters(),lr=1e-4)

    trainer = Trainer(
        diffusion_model=diffusion,
        folder="/projects/data/medical_imaging/duke2/BCS_DBT_Train",
        train_batch_size=128,
        train_lr=1e-4,
        save_and_sample_every=35,
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        optimizer=optimizer,
        gpu_id=local_rank,
        num_epoch=5
    )

    if any(file.endswith(".pt") for file in os.listdir("../results")):
        milestone=trainer.load(-1)
        print(f"Found checkpoint -> model-{milestone}.pt")

    trainer.train()
    destroy_process_group()

if __name__=='__main__':
    main()
