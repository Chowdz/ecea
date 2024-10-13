import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SatelliteDataset(Dataset):
    def __init__(self, img_path, crop_size=5120, patch_size=256, step_size=236, vis_params=None):

        self.file_name = os.path.basename(img_path)

        if not img_path.lower().endswith('.tif'):
            raise ValueError("The input image file is not in TIF format.")

        self.img = Image.open(img_path)
        if self.img.format != 'TIFF':
            raise ValueError("The input image file is not in TIF format.")

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.step_size = step_size

        if self.img.mode != 'RGB':
            raise ValueError("The input image must be in RGB format.")

        if self.img.width < self.crop_size or self.img.height < self.crop_size:
            raise ValueError(
                f"The input image width and height must be greater than {self.crop_size} pixels. Please provide a larger image.")

        self.img = self.center_crop(self.img, self.crop_size)
        self.patches = self.split_into_patches_with_step(self.img, self.patch_size, self.step_size)
        self.vis_params = vis_params
        if self.vis_params is not None:
            self.validate_vis_params(vis_params)

    def validate_vis_params(self, vis_params):
        """Validate that vis_params contains 'min', 'max', 'gamma' keys and all values are numeric."""
        if not isinstance(vis_params, dict):
            raise ValueError("vis_params must be a dictionary.")

        required_keys = ['min', 'max', 'gamma']

        for key in required_keys:
            if key not in vis_params:
                raise ValueError(f"vis_params must contain the key '{key}'.")

            if not isinstance(vis_params[key], (int, float)):
                raise ValueError(f"The value of '{key}' in vis_params must be a numeric type.")

        if len(vis_params) != len(required_keys):
            raise ValueError(f"vis_params must only contain the keys: {', '.join(required_keys)}.")

    def center_crop(self, img, crop_size):
        """Crop a central region of the image with the given size"""
        center_x, center_y = img.width // 2, img.height // 2
        left = center_x - crop_size // 2
        upper = center_y - crop_size // 2
        right = left + crop_size
        lower = upper + crop_size

        return img.crop((left, upper, right, lower))

    def mirror_padding(self, img_tensor, target_h, target_w):
        """Mirror padding for image tensor to reach the target size."""
        C, H, W = img_tensor.shape
        pad_h = max(0, target_h - H)
        pad_w = max(0, target_w - W)

        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return img_tensor

    def split_into_patches_with_step(self, img, patch_size, step_size):
        """Split the image into patches of 256x256 with a step size of 246, and use mirror padding to handle the remaining part."""
        img_tensor = transforms.ToTensor()(img)

        if not torch.any(img_tensor == 0):
            raise ValueError("The input image contains no zero pixels. No inpainting is needed for this image.")

        C, H, W = img_tensor.shape
        patches = []

        # Sliding window with step_size to generate overlapping patches
        for i in range(0, H, step_size):
            for j in range(0, W, step_size):
                patch = img_tensor[:, i:i + patch_size, j:j + patch_size]

                # Handle the case where patch is smaller than patch_size due to reaching the edge
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    patch = self.mirror_padding(patch, patch_size, patch_size)

                patches.append(patch)
        patches = torch.stack(patches)

        return patches

    def apply_vis_params(self, img_tensor):
        """Apply visualization parameters to the image, if provided"""
        if self.vis_params is not None:
            img_tensor = torch.clamp(img_tensor, self.vis_params['min'], self.vis_params['max'])
            img_tensor = (img_tensor - self.vis_params['min']) / (
                    self.vis_params['max'] - self.vis_params['min'])
            img_tensor = img_tensor ** self.vis_params['gamma']
        return img_tensor

    def extract_landsat7_mask(self, image_tensor):
        mask = (image_tensor == 0).any(dim=0).float()
        return mask.unsqueeze(0)

    def extract_cshe_edge(self, image_tensor, mask_tensor):

        img_gray = np.array(transforms.ToPILImage()(image_tensor).convert('L'), dtype=np.float32)
        mask_np = mask_tensor.squeeze().numpy()

        if np.all(mask_np == 0):
            img_gray_canny = img_gray.astype(np.uint8)
        else:
            img_gray_canny = cv2.inpaint(img_gray.astype(np.uint8), mask_np.astype(np.uint8), 1, cv2.INPAINT_TELEA)

        edges_canny = cv2.Canny(img_gray_canny, 50, 120)
        edges_canny = torch.tensor(edges_canny, dtype=torch.float32).unsqueeze(0)

        img_gray[mask_np == 1] = np.nan

        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_z = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)

        edges_sobel = sobel_x ** 2 + sobel_y ** 2 + sobel_z ** 2
        edges_sobel = np.nan_to_num(edges_sobel)
        edges_sobel = torch.tensor(edges_sobel, dtype=torch.float32).unsqueeze(0)
        edges_sobel = torch.sqrt(edges_sobel)

        cshe = 0.5 * edges_canny + 0.5 * edges_sobel
        cshe = (cshe - torch.min(cshe)) / (torch.max(cshe) - torch.min(cshe))

        return cshe

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = self.patches[index]
        patch = self.apply_vis_params(patch)
        mask = self.extract_landsat7_mask(patch)
        cshe = self.extract_cshe_edge(patch, mask)
        return patch, mask, cshe
