import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
from typing import List

import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as calc_ssim
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from .FacialRecognition import FR

def train_protego_mask(cfgs: OmegaConf, 
                        frs: List[FR], 
                        train_dl: DataLoader, 
                        results_save_path: str = None) -> torch.Tensor:
    """
    Train the protection mask using Protego method.

    Args:
        cfgs (OmegaConf): The configuration object containing training parameters.
        frs (List[FR]): List of facial recognition models to use for feature extraction.
        train_dl (DataLoader): The dataloader for training.
        results_save_path (str): The path to save the results. If None, the training loss plots will not be saved.

    Returns:
        torch.Tensor: The trained universal mask of shape [1, 3, mask_size, mask_size].
    """
    # Unpack the parameter configuration
    epoch_num = cfgs.epoch_num
    learning_rate = cfgs.learning_rate
    epsilon = cfgs.epsilon
    min_ssim = cfgs.min_ssim
    mask_size = cfgs.mask_size
    device = frs[0].device
    mask_random_seed = cfgs.mask_random_seed

    # Init the mask
    mask_rand_generator = torch.Generator(device=device)
    mask_rand_generator.manual_seed(mask_random_seed)
    univ_mask = (torch.rand(size=(1, 3, mask_size, mask_size), device=device, requires_grad=False, generator=mask_rand_generator, dtype=torch.float32) - 0.5) * epsilon / 0.5

    # Init the recorders
    losses, hyper_losses, feature_losses, percep_losses = [], [], [], []

    # Init the hyperparameters
    constant = 1e-4 # Used for ensuring numerical stability in the Gram matrix calculation.
    percep_loss_weight = 1. 

    # The training loop
    for epoch in range(epoch_num):
        pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
        batch_losses, batch_hyper_losses, batch_feature_losses, batch_percep_losses = [], [], [], []
        for batch_idx, tensors in pbar:
            orig_faces = tensors[0].to(device)
            uvs = tensors[1].to(device)
            if cfgs.bin_mask:
                bin_masks = tensors[2].to(device)
            #print(bin_masks.shape)
            img_num = orig_faces.shape[0]
            textures = univ_mask.clone().repeat(img_num, 1, 1, 1).requires_grad_(True).to(device)
            perturbations = torch.clamp(F.grid_sample(input=textures, grid=uvs, mode='bilinear', align_corners=True), -epsilon, epsilon)  # (N x 3 x H x W)
            if cfgs.bin_mask:
                restricted_pert = perturbations * bin_masks
                protected_faces = torch.clamp(orig_faces + restricted_pert, 0, 1)
            else:
                protected_faces = torch.clamp(orig_faces + perturbations, 0, 1)

            percep_loss = calc_ssim(protected_faces, orig_faces, data_range=1., size_average=True)
            percep_term = percep_loss_weight * torch.max(min_ssim - percep_loss, torch.tensor(0.).to(device))

            ensemble_losses, _hyper_loss, _feature_loss, _percep_loss = [], 0, 0 ,0
            for fr in frs:
                protected_features = fr(protected_faces)
                normalized_protected_features = F.normalize(protected_features, p=2, dim=1)
                orig_features = fr(orig_faces)
                gram_matrix = torch.matmul(normalized_protected_features, normalized_protected_features.T)  # (N x N)
                regularized_gram_matrix = gram_matrix + constant * torch.eye(gram_matrix.shape[0], device=gram_matrix.device)  # Regularization
                gram_det = torch.linalg.det(regularized_gram_matrix)
                hyper_loss = -torch.log(gram_det + constant)

                feature_loss = F.cosine_similarity(orig_features, protected_features).mean()

                ensemble_losses.append(hyper_loss + feature_loss + percep_term)
                _hyper_loss += hyper_loss.clone().detach()
                _feature_loss += feature_loss.clone().detach()
                _percep_loss += percep_loss.clone().detach()
                
            loss = torch.mean(torch.stack(ensemble_losses, dim=0), dim=0)
            loss.backward()

            # Update the mask
            grad_avg = torch.mean(textures.grad.detach(), dim=0, keepdim=True)
            univ_mask = torch.clamp(univ_mask - learning_rate * torch.sign(grad_avg), -epsilon, epsilon) 

            # Clear the gradients
            textures.grad = None
            loss = loss.detach().cpu().numpy().item()

            # Record the loss
            hyper_loss = _hyper_loss.item() / len(frs) 
            feature_loss = _feature_loss.item() / len(frs) 
            percep_loss = _percep_loss.item() / len(frs) 
            batch_losses.append(loss)
            batch_hyper_losses.append(hyper_loss)
            batch_feature_losses.append(feature_loss)
            batch_percep_losses.append(percep_loss)
            pbar.set_description(f"Epoch {epoch+1}/{epoch_num} | Loss: {loss:.4f} | Hyper Loss: {hyper_loss:.4f} | Feature Loss: {feature_loss:.4f} | Percep Loss: {percep_loss:.4f}")

            if batch_percep_losses[-1] >= min_ssim*1.01 and percep_loss_weight > 1. / 129:
                #print(f"percep_loss_weight decreased from {percep_loss_weight} to {percep_loss_weight/2} at epoch {epoch+1}, batch {batch_idx+1}.")
                percep_loss_weight /= 2
            elif batch_percep_losses[-1] <= min_ssim*0.99 and percep_loss_weight <= 129.:
                #print(f"percep_loss_weight increased from {percep_loss_weight} to {percep_loss_weight*2} at epoch {epoch+1}, batch {batch_idx+1}.")
                percep_loss_weight *= 2
            elif min_ssim*0.99 < batch_percep_losses[-1] < min_ssim*1.01:
                #print(f"percep_loss_weight changed from {percep_loss_weight} to 1 at epoch {epoch+1}, batch {batch_idx+1}.")
                percep_loss_weight = 1.

        # Record the average loss for the epoch
        losses.append(np.mean(batch_losses))
        hyper_losses.append(np.mean(batch_hyper_losses))
        feature_losses.append(np.mean(batch_feature_losses))
        percep_losses.append(np.mean(batch_percep_losses))
        
    if results_save_path is not None:
        # Plot the losses
        plt.figure()
        plt.plot(losses, label='Loss')
        plt.plot(hyper_losses, label='Hyper Loss')
        plt.plot(feature_losses, label='Feature Loss')
        plt.plot(percep_losses, label='Percep Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Losses')
        plt.legend()
        plt.savefig(os.path.join(results_save_path, 'losses.png'))
        #plt.show()
        plt.close()

    return univ_mask.detach().cpu()