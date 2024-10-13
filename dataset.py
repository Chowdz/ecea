import os
import cv2
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SatelliteDateset(Dataset):
    def __init__(self, image_root, mask_root, image_type='image'):
        super(SatelliteDateset, self).__init__()

        self.image_files = glob.glob(image_root + '/*.*')
        self.image_files = sorted(self.image_files)

        self.mask_files = glob.glob(mask_root + '/*.*')
        self.mask_files = sorted(self.mask_files)

        self.number_image = len(self.image_files)
        self.number_mask = len(self.mask_files)
        self.image_type = image_type

        self.shuffle_masks()

    def shuffle_masks(self):
        random.shuffle(self.mask_files)

    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])
        img_gray = np.array(image.convert('L'))

        if self.image_type == 'tensor':
            img_tensor = torch.load(self.image_files[index % self.number_image])
        elif self.image_type == 'image':
            img_tensor = self.image_to_tensor()(image)
        else:
            raise TypeError("The input can only be 'tensor' or 'image'.")

        mask_tensor = self.image_to_tensor()(Image.open(self.mask_files[index % self.number_mask]))
        if not torch.all(mask_tensor == 0):
            mask_tensor = (mask_tensor - torch.min(mask_tensor)) / (torch.max(mask_tensor) - torch.min(mask_tensor))

        edges_canny = cv2.Canny(img_gray, 50, 120)
        edges_canny = torch.tensor(edges_canny, dtype=torch.float32).unsqueeze(0)

        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_z = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)

        edges_sobel = torch.tensor(sobel_x ** 2 + sobel_y ** 2 + sobel_z ** 2, dtype=torch.float32).unsqueeze(0)
        edges_sobel = torch.sqrt(edges_sobel)

        cshe = 0.5 * edges_canny + 0.5 * edges_sobel
        cshe_mask = cshe * (1 - mask_tensor)
        cshe_mask = (cshe_mask - torch.min(cshe_mask)) / (torch.max(cshe_mask) - torch.min(cshe_mask))

        cshe = (cshe - torch.min(cshe)) / (torch.max(cshe) - torch.min(cshe))

        return img_tensor, mask_tensor, cshe_mask, cshe

    def __len__(self):
        return self.number_image

    def image_to_tensor(self):
        return transforms.ToTensor()

    def tensor_to_image(self):
        return transforms.ToPILImage()

    def custom_sort(self, file_path):
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        row = int(parts[0])
        col = int(parts[1].split('.')[0])
        return (row, col)
