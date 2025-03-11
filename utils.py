import torch
import torchvision
from torchvision.transforms import v2 as transforms
import torch.nn.functional as F
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import numpy as np
import random
import lpips
import utils

class Perturbation():
    def __init__(self, data):
        self.data = data
    
    def _attach_black_patch(self, img, x, y, patch_size):
        """
        Attach a black patch (zeros) of given size at position (x, y).
        """
        img_clone = img.clone()  # Clone to avoid modifying original image
        img_clone[:, y:y+patch_size, x:x+patch_size] = 0  # Set pixels to black (0)
        return img_clone

    def _slide_patches(self, image, stride, patch_size):
        frames = []
        for y in range(0, image.shape[1] - 2, stride):
            for x in range(0, image.shape[2] - 2, stride):
                patched_image = self._attach_black_patch(image, x, y, patch_size)
                if (image != patched_image).sum() > patch_size**2/2:
                    frames.append(patched_image)
        return frames
       
    def apply_black_patches(self, stride, patch_size):
        images_perturbed = []
        for image_original, label in tqdm(self.data):
            frames = self._slide_patches(image_original, stride, patch_size)
            try:
                frames = torch.stack(frames)
            except:
                print("Error: Patch size is too big! Reduce patch size or change random seed")
                return None, None
            images_perturbed.append((frames, label))
        return image_original, images_perturbed

def save_png(sample):
    plt.imsave("patched_mnist.png", sample.squeeze(), cmap='gray')

def save_gif(sample):
    imageio.mimsave("mnist_black_patch.gif", np.array(sample.squeeze())*256, fps=10)