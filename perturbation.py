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
subset_indices = random.choices(range(60000), k=1000)  # First 1000 samples
data = Subset(mnist_train, subset_indices)

CF = utils.Perturbation(data)
########## black patches (removing part of digits)
images_perturbed = CF.apply_black_patches(stride=2, patch_size=5) # output size: (1000 x #perturbed_images)
utils.save_png(sample=images_perturbed[20][0][0]) # SEE SAMPLE!!
utils.save_gif(sample=images_perturbed[20][0])

########## white patches (adding new part to digits)
from importlib import reload
reload(utils)
images_perturbed = CF.apply_white_patches(stride=1, patch_size=4) # output size: (1000 x #perturbed_images)