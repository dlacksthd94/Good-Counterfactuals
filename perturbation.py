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

# sample 1000 images
random.seed(123)
subset_indices = random.choices(range(60000), k=100)  # First 1000 samples
data = Subset(mnist_train, subset_indices)

CF = utils.Perturbation(data)
########## black patches (removing part of digits)
images_black_patch = CF.apply_black_patches(stride=2, patch_width=4, patch_height=4) # output size: (1000 x #perturbed_images)
sample_frame = random.choice(images_black_patch)[0]
utils.save_gif(sample_frame, "mnist_black_patch.gif")

########## white patches (adding new part to digits)
images_white_patch = CF.apply_white_patches(stride=2, patch_width=4, patch_height=4) # output size: (1000 x #perturbed_images)
sample_frame = random.choice(images_white_patch)[0]
utils.save_gif(sample_frame, "mnist_white_patch.gif")
