import os
import torch
import gdown
import tempfile
import numpy as np

from torch import nn
from PIL import Image
from options import Options
from models import Generator
from dataset import SatelliteDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def download_model_if_needed(model_url):
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, 'EET.pth')

    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url}...")
        gdown.download(model_url, model_path, quiet=False)
        print(f"Model downloaded and saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

    return model_path


def average_overlap(patch1, patch2, overlap_size=10, direction='horizontal'):
    """
    Calculate the average in the overlapping region of two patches.
    Parameters:
        patch1 (torch.Tensor): First patch (3, 256, 256).
        patch2 (torch.Tensor): Second patch (3, 256, 256).
        overlap_size (int): Number of overlapping pixels.
        direction (str): 'horizontal' for left-right overlap, 'vertical' for top-bottom overlap.
    Returns:
        combined_patch (torch.Tensor): The combined patch with averaged overlapping region.
    """
    if direction == 'horizontal':
        # Take average of the overlapping region on the right of patch1 and left of patch2
        patch1_overlap = patch1[:, :, :, -overlap_size:]
        patch2_overlap = patch2[:, :, :, :overlap_size]
        avg_overlap = (patch1_overlap + patch2_overlap) / 2

        combined_patch = torch.cat([patch1[:, :, :, :-overlap_size], avg_overlap, patch2[:, :, :, overlap_size:]],
                                   dim=3)

    elif direction == 'vertical':
        # Take average of the overlapping region on the bottom of patch1 and top of patch2
        patch1_overlap = patch1[:, :, -overlap_size:, :]
        patch2_overlap = patch2[:, :, :overlap_size, :]
        avg_overlap = (patch1_overlap + patch2_overlap) / 2

        combined_patch = torch.cat([patch1[:, :, :-overlap_size, :], avg_overlap, patch2[:, :, overlap_size:, :]],
                                   dim=2)

    return combined_patch


def stitch_patches(patches, rows_per_image=22, overlap_size=20):
    """
    Stitch patches with overlaps into a large image.
    Parameters:
        patches (torch.Tensor): Tensor of patches with shape (N, 2, 3, 256, 256) where N is the number of patches.
        rows_per_image (int): Number of rows per image.
        overlap_size (int): Size of the overlap in pixels.
    Returns:
        stitched_image (torch.Tensor): The stitched image with size (3, H, W).
    """
    num_patches = patches.shape[0]
    row_patches = []

    # First, stitch each row of patches together horizontally
    for i in range(0, num_patches, rows_per_image):
        row_patch = patches[i]

        for j in range(1, rows_per_image):
            next_patch = patches[i + j]
            row_patch = average_overlap(row_patch, next_patch, overlap_size=overlap_size, direction='horizontal')

        row_patches.append(row_patch)


    # Now stitch the rows together vertically
    stitched_image = row_patches[0]
    for i in range(1, len(row_patches)):
        stitched_image = average_overlap(stitched_image, row_patches[i], overlap_size=overlap_size,
                                         direction='vertical')

    # Trim the rightmost and bottom parts to fit into 5120x5120
    stitched_image = stitched_image[:, :, :5120, :5120]

    return stitched_image


def inpainting_process(input_path, output_path, vis_param=None):
    """
    This function performs inpainting on a landsat-7 ETM+ corrupted satellite image and saves the inpainting image.

    Parameters:
    :param input_path: input_path (str): Path to the input landsat-7 ETM+ corrupted satellite image, specified to the image file.
                      Example: '/home/user/dataset/CorruptedImage.tif'
    :param output_path: output_path (str): Path to the folder where the inpainting satellite image will be saved.
                       The output image will be saved as 'inpainting.tif' and 'ground_truth.tif'.
                       Example: '/home/user/dataset/InpaintingImages/'
    :param vis_param: vis_param (dict, optional): Visualization parameters to adjust the display of the input satellite image.
                                This parameter is not recommended for default use.
                                Only use if the downloaded satellite image has not been adjusted for visualization.
                                Suggested parameters:
                                vis_param = {
                                    'min': 0,
                                    'max': 0.4,
                                    'gamma': 1.4
                                }
    Functionality:
    1. Automatically downloads and loads the pre-trained model.
    2. Loads the landsat-7 ETM+ corrupted satellite image from the specified input path.
    3. Uses a generator model to repair the damaged image.
    4. Saves the inpainting image to the specified output folder.

    Output:
    The inpainting satellite image will be saved in the specified output folder as two files:
    - 'inpainting.tif': The repaired satellite image.
    - 'ground_truth.tif': The original damaged image (for comparison).

    Raises:
    ValueError: If input_path is not a file or output_path is not a directory.
    """

    if not os.path.isfile(input_path):
        raise ValueError(f"The input_path must be a valid file path. Provided: {input_path}")

    if not os.path.isdir(output_path):
        raise ValueError(f"The output_path must be a valid directory. Provided: {output_path}")

    opt = Options()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_url = 'https://drive.google.com/uc?export=download&id=1yon1mfSKmjiEAsK-MTebKZclp1GoZj99'
    model_path = download_model_if_needed(model_url)

    G = Generator(img_size=opt.IMG_SIZE, in_c=opt.IN_C, out_c=opt.OUT_C, patch_size=opt.PATCH_SIZE,
                  embed_dim=opt.EMBED_DIM,
                  depth=opt.DEPTH, num_heads=opt.NUM_HEADS).to(device)
    G.load_state_dict(torch.load(model_path))
    dataset = SatelliteDataset(input_path, vis_params=vis_param)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    patches = []
    with torch.no_grad():
        G.eval()
        for index, x in enumerate(dataloader):
            img_tensor, mask, cshe_edge = x[0].to(device), x[1].to(device), x[2].to(device)
            img_tensor = nn.Parameter(img_tensor, requires_grad=False)
            cshe_edge = nn.Parameter(cshe_edge, requires_grad=False)
            img_fake, cshe_fake = G(img_tensor, cshe_edge, mask)

            real_fake = torch.cat([img_tensor, img_fake], dim=0)
            patches.append(real_fake)

    patches = torch.stack(patches)

    # Stitch patches into the final image
    final_tensor = stitch_patches(patches, rows_per_image=22, overlap_size=20)
    raw_tensor, fake_tensor = final_tensor[0], final_tensor[1]

    name_parts = dataset.file_name.split('.')

    inpainting_img_path = os.path.join(output_path, name_parts[0] + '_Inpainting.' + name_parts[1])
    ground_truth_img_path = os.path.join(output_path, name_parts[0] + '_Corrupted.' + name_parts[1])

    fake_img = transforms.ToPILImage()(fake_tensor)
    fake_img.save(inpainting_img_path)

    raw_img = transforms.ToPILImage()(raw_tensor)
    raw_img.save(ground_truth_img_path)

    # Print success message with output paths
    print(f"Inpainting successful. The result image is saved at: {inpainting_img_path}")
    print(f"The ground truth image is saved at: {ground_truth_img_path}")


if __name__ == '__main__':
    opt = Options()
    args = opt.parse_arguments()
    inpainting_process(input_path=args.INPUT_IMG_PATH, output_path=args.OUTPUT_IMG_PATH, vis_param=args.VIS_PARAM)
