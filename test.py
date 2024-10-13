"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2024/1/12 15:25 
"""
import sys
import os
import torch
import csv
from torch import nn
from options import Options
from models import Generator
from dataset import SatelliteDateset
from torch.utils.data import DataLoader
from utils import psnr_value, ssim_value, mape_value, tensor_to_image

opt = Options()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inpainting_generator = Generator(img_size=opt.IMG_SIZE, in_c=opt.IN_C, out_c=opt.OUT_C, patch_size=opt.PATCH_SIZE,
                                 embed_dim=opt.EMBED_DIM, depth=opt.DEPTH, num_heads=opt.NUM_HEADS).to(device)
inpainting_generator.load_state_dict(torch.load(r'E:/Dataset/spacenet/save_model/G_1030.pth'))

dataset_test = SatelliteDateset(image_root='E:/Dataset/spacenet/test', mask_root='E:/Dataset/spacenet/40-50%')
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

with torch.no_grad():
    inpainting_generator.eval()
    for index, x in enumerate(dataloader_test):
        img_truth, mask, sobel_mask = x[0].to(device), x[1].to(device), x[2].to(device)
        img = nn.Parameter(img_truth * (1 - mask), requires_grad=False)
        sobel_mask = nn.Parameter(sobel_mask, requires_grad=False)
        img_fake, sobel_fake = inpainting_generator(img, sobel_mask, mask)

        # real_img = torch.split(img_truth, split_size_or_sections=1, dim=0)
        # real_img_comp = torch.cat(real_img, dim=3).reshape(3, 256, -1)
        #
        # real_miss = torch.split(img + mask, split_size_or_sections=1, dim=0)
        # real_miss_comp = torch.cat(real_miss, dim=3).reshape(3, 256, -1)
        #
        # fake_sobel = torch.split(sobel_fake, split_size_or_sections=1, dim=0)
        # fake_sobel_comp = torch.cat(fake_sobel, dim=3).reshape(1, 256, -1).repeat(3, 1, 1)

        # fake_img = torch.split(img_fake, split_size_or_sections=1, dim=0)
        # fake_img_comp = torch.cat(fake_img, dim=3).reshape(3, 256, -1)

        # comp = torch.cat([real_img_comp, real_miss_comp, fake_img_comp], dim=1)
        comp_pic = tensor_to_image(img_fake.squeeze(0))
        comp_pic.save('E:/Dataset/spacenet/fid_50/' + str(index) + '.png')

