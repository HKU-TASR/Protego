import os
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from protego.UVMapping import UVGenerator
from protego.FaceDetection import FD
from protego.utils import load_imgs, load_mask, crop_face, complete_del
from protego import BASE_PATH

def replic_attack(uncropped_imgs: List[torch.Tensor], 
                  mask: torch.Tensor, 
                  rep_mask: torch.Tensor, 
                  epsilon: float, 
                  three_d: bool, 
                  bin_mask: bool, 
                  fd: FD, 
                  uvmapper: UVGenerator, 
                  device: torch.device, 
                  resize_face: bool) -> Dict[str, List[torch.Tensor]]:
    resized_faces, faces, positions = [], [], []
    for img_idx, raw_img in tqdm.tqdm(enumerate(uncropped_imgs), desc="Detecting and cropping faces from raw images"):
        raw_img = raw_img.to(device).squeeze(0)
        face, pos = crop_face(raw_img, fd, verbose=False, strict=False)
        if face is None or pos is None:
            print(f"Warning: No valid face detected in image {img_idx}.")
            continue
        pos = list(pos)
        pos.append(img_idx)
        resized_faces.append(F.interpolate(face.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0))
        if not resize_face:
            faces.append(face)
        positions.append(pos)
    resized_faces = torch.stack(resized_faces, dim=0)
    uvs, bin_masks, _ = uvmapper.forward(imgs=resized_faces)
    perts = torch.clamp(F.grid_sample(mask.repeat(resized_faces.shape[0], 1, 1, 1), uvs, mode='bilinear', align_corners=True), -epsilon, epsilon) if three_d else mask.repeat(faces.shape[0], 1, 1, 1)
    if bin_mask:
        perts = perts * bin_masks
    del uvs, bin_masks
    complete_del()
    protected_imgs, orig_imgs, vis_perts = [], [], []
    for pos_idx, pos in enumerate(positions):
        x1, y1, x2, y2, img_idx = pos
        protected_img = uncropped_imgs[img_idx].clone().squeeze(0)
        #print(protected_img.shape, protected_img.min(), protected_img.max())
        if resize_face:
            pert = perts[pos_idx]
            protected_face = torch.clamp(resized_faces[pos_idx] + pert, 0, 1)
            protected_face = F.interpolate(protected_face.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
            protected_img[:, y1:y2, x1:x2] = protected_face
        else:
            pert = F.interpolate(perts[[pos_idx]], size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
            protected_face = torch.clamp(faces[pos_idx] + pert, 0, 1)
            protected_img[:, y1:y2, x1:x2] = protected_face
        vis_pert = torch.zeros_like(protected_img)
        if pert.size(1) != (y2 - y1) or pert.size(2) != (x2 - x1):
            pert = F.interpolate(pert.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
        vis_pert[:, y1:y2, x1:x2] = pert
        protected_img = protected_img
        protected_imgs.append(protected_img)
        orig_imgs.append(uncropped_imgs[img_idx].squeeze(0).detach().cpu())
        vis_perts.append(vis_pert)
    protected_faces, have_face, positions = [], [], []
    for img_idx, protected_img in tqdm.tqdm(enumerate(protected_imgs), desc="Re-detecting and cropping faces from protected images"):
        protected_face, pos = crop_face(protected_img, fd, verbose=False, strict=False)
        if protected_face is None or pos is None:
            print(f"Warning: No valid face detected in protected image {img_idx}.")
            continue
        protected_faces.append(F.interpolate(protected_face.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0))
        have_face.append(img_idx)
        positions.append(pos)
    protected_faces = torch.stack(protected_faces, dim=0)
    orig_imgs = [orig_imgs[i] for i in have_face]
    protected_imgs = [protected_imgs[i] for i in have_face]
    perts = [vis_perts[i] for i in have_face]
    uvs, bin_masks, _ = uvmapper.forward(imgs=protected_faces)
    rep_perts = torch.clamp(F.grid_sample(rep_mask.repeat(protected_faces.shape[0], 1, 1, 1), uvs, mode='bilinear', align_corners=True), -epsilon, epsilon) if three_d else rep_mask.repeat(protected_faces.shape[0], 1, 1, 1)
    if bin_mask:
        rep_perts = rep_perts * bin_masks
    del uvs, bin_masks
    complete_del()
    attacked_imgs, vis_rep_perts = [], []
    for pos_idx, pos in enumerate(positions):
        x1, y1, x2, y2 = pos
        attacked_img = protected_imgs[pos_idx].clone()
        rep_pert = rep_perts[pos_idx]
        attacked_face = torch.clamp(protected_faces[pos_idx] - rep_pert, 0, 1)
        attacked_face = F.interpolate(attacked_face.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
        attacked_img[:, y1:y2, x1:x2] = attacked_face
        vis_rep_pert = torch.zeros_like(attacked_img)
        if rep_pert.size(1) != (y2 - y1) or rep_pert.size(2) != (x2 - x1):
            rep_pert = F.interpolate(rep_pert.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
        vis_rep_pert[:, y1:y2, x1:x2] = rep_pert
        attacked_imgs.append(attacked_img)
        vis_rep_perts.append(vis_rep_pert)
    return {
        'orig': orig_imgs,
        'prot': protected_imgs,
        'repatt': attacked_imgs, 
        'pert': perts,
        'rep_pert': vis_rep_perts
    }

def visualize(res: Dict[str, List[torch.Tensor]], save_dir: str):
    for idx, tensors in enumerate(zip(res['orig'], res['prot'], res['repatt'], res['pert'], res['rep_pert'])):
        orig_img, prot_img, repatt_img, pert, rep_pert = tensors
        orig_img = orig_img.permute(1, 2, 0).mul(255.).detach().cpu().numpy().astype(np.uint8)
        prot_img = prot_img.permute(1, 2, 0).mul(255.).detach().cpu().numpy().astype(np.uint8)
        repatt_img = repatt_img.permute(1, 2, 0).mul(255.).detach().cpu().numpy().astype(np.uint8)
        pert = pert
        rep_pert = rep_pert
        pert_diff = pert - rep_pert
        pert = ((pert - pert.min()) / (pert.max() - pert.min())).permute(1, 2, 0).mul(255.).detach().cpu().numpy().astype(np.uint8)
        rep_pert = ((rep_pert - rep_pert.min()) / (rep_pert.max() - rep_pert.min())).permute(1, 2, 0).mul(255.).detach().cpu().numpy().astype(np.uint8)
        pert_diff = ((pert_diff - pert_diff.min()) / (pert_diff.max() - pert_diff.min())).permute(1, 2, 0).mul(255.).detach().cpu().numpy().astype(np.uint8)
        plt.figure(figsize=(15, 9))
        plt.subplot(2, 3, 1)
        plt.title('Original Image')
        plt.imshow(orig_img)
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.title('Replication Attacked Image')
        plt.imshow(repatt_img)
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.title('Protected Image')
        plt.imshow(prot_img)
        plt.axis('off')
        plt.subplot(2, 3, 4)
        plt.title('Perturbation')
        plt.imshow(pert)
        plt.axis('off')
        plt.subplot(2, 3, 6)
        plt.title('Replication Perturbation')
        plt.imshow(rep_pert)
        plt.axis('off')
        plt.subplot(2, 3, 5)
        plt.title('Perturbation Difference (Pert - Rep Pert)')
        plt.imshow(pert_diff)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'visualization_{idx:04d}.png'))
        plt.close()

if __name__ == '__main__':
    with torch.no_grad():
        device = torch.device('cuda:7')
        protectee = "Bradley_Cooper"
        uncropped_imgs = load_imgs(os.path.join(BASE_PATH, 'face_db', 'fs_uncropped', protectee),
                                img_paths=None, img_sz=-1, usage_portion=1., drange=1, device=device, return_img_paths=False)
        mask = load_mask(os.path.join(BASE_PATH, 'experiments', 'default', protectee, 'univ_mask.npy'), device=device)
        rep_mask = load_mask(os.path.join(BASE_PATH, 'experiments', 'default', protectee, 'univ_mask.npy'), device=device)
        fd = FD(model_name='mobilenet_retinaface_widerface', device=device)
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        
        res = replic_attack(uncropped_imgs=uncropped_imgs, 
                            mask=mask, 
                            rep_mask=rep_mask, 
                            epsilon=16./255.,
                            three_d=True, 
                            bin_mask=True, 
                            fd=fd,
                            uvmapper=uvmapper,
                            device=device,
                            resize_face=False)
        save_path = f'{BASE_PATH}/trash/rep_att_vis/repatt_completeleak'
        os.makedirs(save_path, exist_ok=True)
        visualize(res, save_path)
