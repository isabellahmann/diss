# copied for the moment

from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
import matplotlib
import wandb

import torch.nn.functional as F
from test_copied_refactoring import SupnBRATS
from supn_base.supn_distribution import SUPN

matplotlib.use('Agg')
def sample_validate_colour(image_path, model=SupnBRATS(), nr_of_samples=10, log_wandb=False, train_set=False):
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch


    # def normalize(image_tensor):
    #         """Normalize the image tensor."""
    #         normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    #         return normalize(image_tensor)

    image = Image.open(image_path).convert("L")
    image = transforms.ToTensor()(image)

    image = image.unsqueeze(0)

    print(image.size)
    plt.figure(figsize=(6, 6))
    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis('off')
    plt.savefig('original_image.png')


    perceptual_model = SupnBRATS()
    print('loadeed model')

    # img1_tensor, c = preprocess_image(image, size=perceptual_model.image_size)
    # print(img1_tensor.shape,img1_tensor.min().item(),img1_tensor.max().item())
    # print(c.shape,c.min().item(),c.max().item())
    # c = normalize(c)
    model_outputs = perceptual_model.run_model(image)  # Get model outputs
    supn_list = model_outputs[0]

    rows = nr_of_samples  # One row per sample
    cols = 1  # One column per resolution level

    plt.figure(figsize=(cols * 3, rows * 3))

    supn_dist = supn_list

    # Sample from the distribution
    sample = supn_dist.sample(num_samples=nr_of_samples).squeeze(1)
    sample_np = sample.detach().cpu().numpy()

    fig1, axes = plt.subplots(2, 2, figsize=(10, 10))
    vmin, vmax = -1, 1  # Set consistent color scaling limits
    mean = supn_dist.mean.detach().cpu()[:,0,:,:]
    print(mean.shape,mean.min().item(),mean.max().item())
    #print(image.min().item())
    axes[0, 0].imshow(image.squeeze(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    print(supn_dist.mean.detach().cpu().squeeze().shape)
    print(supn_dist.mean.detach().cpu().squeeze()[0].shape)

    mean  = torch.sigmoid(supn_dist.mean)


    # axes[0, 1].imshow(supn_dist.mean.detach().cpu().squeeze()[0], cmap='gray')
    axes[0, 1].imshow(mean.detach().cpu().squeeze(), cmap='gray')
    axes[0, 1].set_title('Mean Reconstruction')
    axes[0, 1].axis('off')

    # axes[1, 0].imshow(-supn_dist.mean.detach().cpu().squeeze()[0] + sample_np[0, 0, :, :], cmap='gray')
    axes[1, 0].imshow(-mean.detach().cpu().squeeze() + sample_np[0, 0, :, :], cmap='gray')
    axes[1, 0].set_title('Sampled Image (No Mean)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sample[0, 0, :, :].detach().cpu(), cmap='gray')
    axes[1, 1].set_title('Mean + Sample')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    if log_wandb:
        if train_set:
            wandb.log({'Train_Recon': wandb.Image(fig1)})
        else:
            wandb.log({'Test_Recon': wandb.Image(fig1)})

    plt.tight_layout()
    plt.savefig('colour_sample.png')


sample_validate_colour("data/flair_images/patient_0_4.png")
