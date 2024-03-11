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
    ).cuda()

    diffusion=GaussianDiffusion(
        model,
        image_size=64,
        num_slices=8,
        timesteps=1000,   # number of steps
        loss_type='l1',    # L1 or L2
        gpu_id=local_rank
    ).cuda()

    train_dataset=Duke_DBT_Dataset("/projects/data/medical_imaging/duke2/Resized64")
    optimizer=Adam(diffusion.parameters(),lr=1e-4)

    trainer = Trainer(
        diffusion,
        "/projects/data/medical_imaging/duke2/Resized64",
        train_batch_size=16,
        train_lr=1e-4,
        save_and_sample_every=35,
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        optimizer=optimizer,
        gpu_id=local_rank,
        num_epoch=50
    )

    if any(file.endswith(".pt") for file in os.listdir("../DDP_results")):
        milestone=trainer.load(-1)
        print(f"Found checkpoint -> model-{milestone}.pt")

    trainer.train()
    destroy_process_group()

if __name__=='__main__':
    main()
