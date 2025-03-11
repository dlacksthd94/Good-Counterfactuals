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
    
    def _attach_black_patch(self, image, x, y, patch_size):
        """
        Attach a black patch (zeros) of given size at position (x, y).
        """
        image_clone = image.clone()  # Clone to avoid modifying original image
        image_clone[:, y:y+patch_size, x:x+patch_size] = 0  # Set pixels to black (0)
        return image_clone

    def _slide_patches(self, image, stride, patch_size):
        frames = []
        for y in range(0, image.shape[1] - 2, stride):
            for x in range(0, image.shape[2] - 2, stride):
                patched_image = self._attach_black_patch(image, x, y, patch_size)
                if (image != patched_image).sum() > patch_size**2/2:
                    frames.append(patched_image)
        return frames

    
    def apply_black_patches(self, stride, patch_size):
        """
        Apply black patches on digit area
        """
        images_perturbed = []
        for image, label in tqdm(self.data):
            frames = self._slide_patches(image, stride, patch_size)
            try:
                frames = torch.stack(frames)
            except:
                print("Error: Patch size is too big! Reduce patch size or change random seed")
                return None, None
            images_perturbed.append((frames, label))
        return images_perturbed
    
    
def save_png(sample):
    plt.imsave("patched_mnist.png", sample.squeeze(), cmap='gray')

def save_gif(sample):
    imageio.mimsave("mnist_black_patch.gif", np.array(sample.squeeze())*256, fps=10)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # sample 1000 images
    random.seed(123)
    subset_indices = random.choices(range(60000), k=1000)  # First 1000 samples
    data = Subset(mnist_train, subset_indices)

    CF = utils.Perturbation(data)
    self = CF
    stride = 1
    patch_size = 4