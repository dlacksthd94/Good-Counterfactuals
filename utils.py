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
        self.std = 1
    
    def _Gaussian_kernel_2D(self, patch_width, patch_height):
        grid_width = torch.linspace(-1, 1, patch_width)
        grid_height = torch.linspace(-1, 1, patch_height)
        D1, D2 = torch.meshgrid(grid_height, grid_width, indexing='ij')
        gaussian_weights = torch.exp(-(D1**2 + D2**2) / (2 * self.std**2))
        noise = torch.randn((patch_height, patch_width)) * self.std * 0.1  # Adding small noise for randomness
        kernel = gaussian_weights + noise
        kernel = kernel - kernel.min()  # Normalize
        kernel = kernel / kernel.max()
        return kernel

    def _attach_black_patch(self, image, x, y, patch_width, patch_height):
        """
        Attach a black patch (zeros) of given size at position (x, y).
        """
        kernel = self._Gaussian_kernel_2D(patch_width, patch_height)
        image_clone = image.clone()  # Clone to avoid modifying original image
        image_clone[:, y:y+patch_height, x:x+patch_width] -= kernel  # Set pixels to black (0)
        image_clone = torch.clamp(image_clone, 0, 1)
        return image_clone
        
    def _attach_white_patch(self, image, x, y, patch_width, patch_height):
        """
        Attach a black patch (zeros) of given size at position (x, y).
        """
        kernel = self._Gaussian_kernel_2D(patch_width, patch_height)
        image_clone = image.clone()  # Clone to avoid modifying original image
        image_clone[:, y:y+patch_height, x:x+patch_width] += kernel  # Set pixels to white (1)
        image_clone = torch.clamp(image_clone, 0, 1)
        return image_clone

    def _slide_black_patches(self, image, stride, patch_width, patch_height):
        images_perturbed = []
        for y in range(0, image.shape[1] - (patch_height - 1), stride):
            for x in range(0, image.shape[2] - (patch_width - 1), stride):
                patched_image = self._attach_black_patch(image, x, y, patch_width, patch_height)
                if (image != patched_image).sum() > patch_height*patch_width/2:
                # if (image.bool() & (image != patched_image)).sum() > 0:
                    images_perturbed.append(patched_image)
        return images_perturbed
    
    def _slide_white_patches(self, image, stride, patch_width, patch_height):
        images_perturbed = []
        for y in range(0, image.shape[1] - (patch_height - 1), stride):
            for x in range(0, image.shape[2] - (patch_width - 1), stride):
                patched_image = self._attach_white_patch(image, x, y, patch_width, patch_height)
                if (image.bool() & (image != patched_image)).sum() > 0:
                    images_perturbed.append(patched_image)
        return images_perturbed
    
    def apply_black_patches(self, stride, patch_width, patch_height):
        """
        Apply black patches on digit area
        """
        result = []
        for image, label in tqdm(self.data):
            images_perturbed = self._slide_black_patches(image, stride, patch_width, patch_height)
            try:
                images_perturbed = torch.stack(images_perturbed)
            except:
                print("Error: Patch size is too big! Reduce patch size or change random seed")
                return None, None
            result.append((image, images_perturbed, label))
        return result
    
    def apply_white_patches(self, stride, patch_width, patch_height):
        """
        Apply white patches on non-digit area
        """
        result = []
        for image, label in tqdm(self.data):
            images_perturbed = self._slide_white_patches(image, stride, patch_width, patch_height)
            try:
                images_perturbed = torch.stack(images_perturbed)
            except:
                print("Error: Patch size is too big! Reduce patch size or change random seed")
                return None, None
            result.append((image, images_perturbed, label))
        return result
    
def save_png(sample, file_name):
    plt.imsave(file_name, sample.squeeze(), cmap='gray')

def save_gif(sample, file_name):
    imageio.mimsave(file_name, np.array(sample)*256, fps=5)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # sample 1000 images
    # random.seed(123)
    subset_indices = random.choices(range(60000), k=10)  # First 1000 samples
    data = Subset(mnist_train, subset_indices)

    CF = Perturbation(data)
    self = CF
    stride = 2
    patch_width = 4
    patch_height = 4

    result_black_patch = CF.apply_black_patches(stride, patch_width, patch_height) # output size: (1000 x #perturbed_images)
    image_original, images_black_patch, label = random.choice(result_black_patch)
    save_gif(images_black_patch.squeeze(), file_name="mnist_black_patch.gif")

    result_white_patch = CF.apply_white_patches(stride, patch_width, patch_height) # output size: (1000 x #perturbed_images)
    image_original, images_white_patch, label = random.choice(result_white_patch)
    save_gif(images_white_patch.squeeze(), file_name="mnist_white_patch.gif")