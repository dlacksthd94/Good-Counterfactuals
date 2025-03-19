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
import pickle

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# sample images
N_SAMPLE = 1000
random.seed(1234)
subset_indices = random.choices(range(60000), k=N_SAMPLE)  # First N samples
data = Subset(mnist_train, subset_indices)

PTB = utils.Perturbation(data)

########## black patches (removing part of digits)
stride = 2
patch_black_width = 8
patch_black_height = 8
result_black_patch = PTB.apply_black_patches(stride, patch_black_width, patch_black_height) # output size: (1000 x #perturbed_images)
image_original, images_black_patch, label = random.choice(result_black_patch)
utils.save_gif(images_black_patch.squeeze(), file_name="mnist_black_patch.gif")
utils.save_png(image_original, file_name="image.png")
with open(f"data/test_black_patch_w{patch_black_width}h{patch_black_height}.pickle", 'wb') as f:
    pickle.dump(result_black_patch, f)

########## black patches (removing part of digits)
stride = 2
patch_white_width = 4
patch_white_height = 4
result_white_patch = PTB.apply_white_patches(stride, patch_white_width, patch_white_height) # output size: (1000 x #perturbed_images)
image_original, images_white_patch, label = random.choice(result_white_patch)
utils.save_gif(images_white_patch.squeeze(), file_name="mnist_white_patch.gif")
utils.save_png(image_original, file_name="image.png")
with open(f"data/test_white_patch_w{patch_white_width}h{patch_white_height}.pickle", 'wb') as f:
    pickle.dump(result_white_patch, f)
