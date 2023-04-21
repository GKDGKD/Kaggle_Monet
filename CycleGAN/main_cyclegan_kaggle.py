import torch
import pandas as pd
import os, sys
import time
import cv2 as cv
from PIL import Image
import numpy as np
import random
import itertools
import datetime
import argparse
import logging

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from log.logutil import Logger
from datasets import ImageDataset
from model_kaggle import Generator, Discriminator, weights_init_normal
from utils import ReplayBuffer, LambdaLR


parser = argparse.ArgumentParser()
parser.add_argument("--epoch_start", type=int, default=0, help="epoch to start training from")
# parser.add_argument('--device', type=str, default='cuda:1', help='device to train model, cpu or cuda')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=1E-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()

save_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
save_dir = f'{os.getcwd()}/result/runs_{save_time}/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

print(f'save_dir = {save_dir}')

logger = Logger(log_id='main',\
                # log_name=f'log_test_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.log', \
                log_name=f'CycleGAN_kaggle.log', \
                log_dir=save_dir).logger
logger.setLevel(logging.DEBUG)
# logger.set_sub_logger('wheat_test')

logger.info(opt)

photo_path = '../data/photo_jpg'
monet_path = '../data/monet_jpg'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #normalize效果特别好
    # transforms.Lambda(lambda x: (x / 127.5) - 1)
    # # 无normalize：
    # # Mean: tensor([0.4038, 0.4086, 0.3840])
    # # Std: tensor([0.2219, 0.2027, 0.2198])

    # # 有normalize:
    # # Mean: tensor([-0.1924, -0.1828, -0.2319])
    # # Std: tensor([0.4438, 0.4055, 0.4396])
])

def denorm(x):
    # out = (x + 1) / 2
    out = x * 0.5 + 0.5
    return torch.clamp(out, 0, 1)

ds = ImageDataset(monet_path, photo_path, transform)
dataloader = DataLoader(ds, opt.batch_size, shuffle=True)
logger.info(f'BATCH_SZIE: {opt.batch_size}, len(dataset): {len(ds)}, len(dataloader_photo): {len(dataloader)}')
logger.info(f"ds[0]['monet'].shape: {ds[0]['monet'].shape}")

img_shape = tuple(ds[0]['photo'].shape)
logger.info(f'img_shape: {img_shape}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device: {device}')
logger.info('Buliding models...')

# Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Initialize generator and discriminator
G_photo2monet = Generator()
G_monet2photo = Generator()
D_photo = Discriminator()
D_monet = Discriminator()

if torch.cuda.is_available():
    G_photo2monet.to(device)
    G_monet2photo.to(device)
    D_monet.to(device)
    D_photo.to(device)

if opt.epochs != 0:
    G_photo2monet.apply(weights_init_normal)
    G_monet2photo.apply(weights_init_normal)
    D_photo.apply(weights_init_normal)
    D_monet.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(
    itertools.chain(G_photo2monet.parameters(), G_monet2photo.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_monet = torch.optim.Adam(D_monet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_photo = torch.optim.Adam(D_photo.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.epochs, opt.epoch_start, opt.decay_epoch).step
)
lr_scheduler_D_photo = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_photo, lr_lambda=LambdaLR(opt.epochs, opt.epoch_start, opt.decay_epoch).step
)
lr_scheduler_D_monet = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_monet, lr_lambda=LambdaLR(opt.epochs, opt.epoch_start, opt.decay_epoch).step
)

# Buffers of previously generated samples
fake_photo_buffer = ReplayBuffer()
fake_monet_buffer = ReplayBuffer()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def sample_images(batches_done, save_path):
    """Saves a generated sample from the test set"""
    imgs = next(iter(dataloader))
    G_photo2monet.eval()
    G_monet2photo.eval()
    real_A = Variable(imgs["photo"].type(Tensor)).to(device)
    fake_B = G_photo2monet(real_A)
    real_B = Variable(imgs["monet"].type(Tensor)).to(device)
    fake_A = G_monet2photo(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(denorm(fake_A), nrow=5, normalize=True)
    fake_B = make_grid(denorm(fake_B), nrow=5, normalize=True)
    logger.info('real_A.shape: {real_A.shape}, fake_A.shape: {fake_A.shape}')
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, f"{save_dir}{batches_done}.png", normalize=False)

os.makedirs(f'{save_dir}/pictures')  
os.makedirs(f'{save_dir}/saved_models')
logger.info('Models are bulited. Starts training...')

prev_time = time.time()
for epoch in range(opt.epochs):
    for i, batch in enumerate(dataloader):
        
        # Set model input
        real_photo = batch['photo'].to(device)
        real_monet = batch['monet'].to(device)
        
        # Adversarial ground truths
        valid = torch.ones((real_photo.size(0), *D_photo.output_shape)).to(device)
        fake = torch.zeros((real_photo.size(0), *D_photo.output_shape)).to(device)

        # ------------------
        #  Train Generators
        # ------------------
        
        G_photo2monet.train()
        G_monet2photo.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_photo = criterion_identity(G_monet2photo(real_photo), real_photo)
        loss_id_monet = criterion_identity(G_photo2monet(real_monet), real_monet)
        loss_identity = (loss_id_photo + loss_id_monet) / 2

        # GAN loss
        fake_monet = G_photo2monet(real_photo)
        fake_photo = G_monet2photo(real_monet)
        valid_G = torch.zeros_like(fake_monet).to(device)
        loss_gan_photo2monet = criterion_GAN(fake_monet, valid_G)
        loss_gan_monet2photo = criterion_GAN(fake_photo, valid_G)
        loss_gan = (loss_gan_monet2photo + loss_gan_photo2monet) / 2
        
        # Cycle loss
        recov_photo = G_monet2photo(fake_monet)
        recov_monet = G_photo2monet(fake_photo)
        loss_cycle_photo = criterion_cycle(recov_photo, real_photo)
        loss_cycle_monet = criterion_cycle(recov_monet, real_monet)
        loss_cycle = (loss_cycle_monet + loss_cycle_photo) / 2

        # Total loss
        loss_G = loss_gan + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator photo
        # -----------------------
        optimizer_D_photo.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_photo(real_photo), valid)
        # Fake loss (on batch of previously generated samples)
        fake_photo_ = fake_photo_buffer.push_and_pop(fake_photo)
        loss_fake = criterion_GAN(D_photo(fake_photo_.detach()), fake)
        # Total loss
        loss_D_photo = (loss_real + loss_fake) / 2
        loss_D_photo.backward()
        optimizer_D_photo.step()

        # -----------------------
        #  Train Discriminator monet
        # -----------------------
        optimizer_D_monet.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_monet(real_monet), valid)
        # Fake loss (on batch of previously generated samples)
        fake_monet_ = fake_monet_buffer.push_and_pop(fake_monet)
        loss_fake = criterion_GAN(D_monet(fake_monet_.detach()), fake)
        # Total loss
        loss_D_monet = (loss_real + loss_fake) / 2
        loss_D_monet.backward()
        optimizer_D_monet.step()

        loss_D = (loss_D_monet + loss_D_photo) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        logger.info(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                opt.epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_gan.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, save_path)

        # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_photo.step()
    lr_scheduler_D_monet.step()

logger.info('Training is done. Saving the models...')
# Save model checkpoints
torch.save(G_photo2monet.state_dict(), f"{save_dir}/saved_models/G_photo2monet.pth")
torch.save(G_monet2photo.state_dict(), f"{save_dir}/saved_models/G_monet2photo.pth")
torch.save(D_photo.state_dict(), f"{save_dir}/saved_models/D_photo.pth")
torch.save(D_monet.state_dict(), f"{save_dir}/saved_models/D_monet.pth")

logger.info('All is done.')