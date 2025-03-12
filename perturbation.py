import torch
import torchvision
from torchvision.transforms import v2 as transforms
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
import random
import lpips
import utils

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# sample images
N_SAMPLE = 100
subset_indices = random.choices(range(60000), k=N_SAMPLE)  # First N samples
data = Subset(mnist_train, subset_indices)

CF = utils.Perturbation(data)
########## black patches (removing part of digits)
result_black_patch = CF.apply_black_patches(stride=2, patch_width=5, patch_height=5) # output size: (1000 x #perturbed_images)
image_original, images_black_patch, label = random.choice(result_black_patch)
utils.save_gif(images_black_patch.squeeze(), file_name="mnist_black_patch.gif")
utils.save_png(image_original, "iamge.png")

########## white patches (adding new part to digits)
result_white_patch = CF.apply_white_patches(stride=2, patch_width=8, patch_height=4) # output size: (1000 x #perturbed_images)
image_original, images_white_patch, label = random.choice(result_white_patch)
utils.save_gif(images_white_patch.squeeze(), file_name="mnist_white_patch.gif")
utils.save_png(image_original, "iamge.png")