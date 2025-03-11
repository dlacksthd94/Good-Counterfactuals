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

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# sample 1000 images
random.seed(123)
subset_indices = random.choices(range(60000), k=1000)  # First 1000 samples
data = Subset(mnist_train, subset_indices)

from importlib import reload
reload(utils)
CF = utils.Perturbation(data)
########## black patches (removing part of digits)
image_original, images_perturbed = CF.apply_black_patches(stride=2, patch_size=5)
utils.save_png(sample=images_perturbed[10][0][0])
utils.save_gif(sample=images_perturbed[10][0])

########## white patches (adding new part to digits)