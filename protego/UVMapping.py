import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
from typing import Tuple, Optional, List

import torch
import torch.nn.functional as F
import cv2
from skimage.transform import estimate_transform, warp
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

from smirk.src.smirk_encoder import SmirkEncoder
from smirk.src.FLAME.FLAME import FLAME
from .FlameRenderer import Renderer

def init_smirk(smirk_ckpts_path: str, device: torch.device) -> SmirkEncoder:
    """
    Initialize the SMIRK encoder with the provided checkpoint path and device.

    Args:
        smirk_ckpts_path (str): Path to the SMIRK checkpoint file.
        device (torch.device): The device to load the model onto (e.g., 'cuda' or 'cpu').
    """
    smirk_encoder = SmirkEncoder().to(device)
    ckpts = torch.load(smirk_ckpts_path, map_location=device, weights_only=False)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in ckpts.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()
    return smirk_encoder

def init_flame(smirk_base_path: str, device: torch.device) -> FLAME:
    return FLAME(base_path=smirk_base_path).to(device)

def init_renderer(smirk_base_path: str, device: torch.device) -> Renderer:
    return Renderer(smirk_path=smirk_base_path).to(device)

def init_mp_lmk_detector(model_asset_path: str = 'assets/face_landmarker.task') -> FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1,
                                        min_face_detection_confidence=0.1,
                                        min_face_presence_confidence=0.1
                                        )
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def run_mediapipe(img_np: np.ndarray, detector: FaceLandmarker) -> Optional[np.ndarray]:
    # print(image.shape)
    #image_numpy = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)
    if len(detection_result.face_landmarks) == 0:
        #print('No face detected')
        return None
    face_landmarks = detection_result.face_landmarks[0]
    face_landmarks_numpy = np.zeros((478, 3))
    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]
    return face_landmarks_numpy

def estimate_tform(landmarks: np.ndarray, h: int, w: int, scale: int = 1.0, image_size: int = 224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform

def restore(tranformed: np.ndarray, tforms: np.ndarray, orig_sz: int) -> np.ndarray:
    """
    Use the transformation matrices to restore the UV maps to the original images' space.

    Args:
        tranformed (np.ndarray): The transformed UV maps in the shape of (N, H, W, C) or (N, H, W, C).
        tforms (np.ndarray): The transformation matrices in the shape of (N, 3, 3).
        orig_sz (int): The original size of the images.
    
    Returns:
        np.ndarray: The restored UV maps in the shape of (N, orig_sz, orig_sz, C).
    """
    num = tranformed.shape[0]
    restoreds = []
    for i in range(num):
        transed = tranformed[i]
        tform = tforms[i]
        restored = warp(transed, tform, output_shape=(orig_sz, orig_sz), preserve_range=True)
        restoreds.append(restored)
        #print(restored.shape, orig_sz, orig_sz)
    restoreds = np.stack(restoreds, axis=0)
    return restoreds

def gen_uvs(imgs: torch.Tensor, smirk_encoder: SmirkEncoder, flame: FLAME, renderer: Renderer, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The function generates UV coordinates for the input images. 

    Args:
        imgs (torch.Tensor): Input images in the shape of (B, 3, 224, 224), RGB, range [0, 1], dtype torch.float32.
        smirk_encoder (SmirkEncoder): The SMIRK encoder model.
        flame (FLAME): The FLAME model.
        renderer (Renderer): The renderer model.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu'). Should be the same as the device used for the models.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - uv_grid (torch.Tensor): UV coordinates in the shape of (B, H, W, 2), where H and W are the height and width of the rendered image.
            - visibility_mask (torch.Tensor): Visibility mask in the shape of (B, H, W, 1), indicating which pixels are visible.
    """
    imgs = imgs.to(device)
    smirk_outputs = smirk_encoder(imgs)
    flame_outputs = flame.forward(smirk_outputs)
    renderer_output = renderer.forward(flame_outputs['vertices'], smirk_outputs['cam'],
                                        landmarks_fan=flame_outputs['landmarks_fan'], landmarks_mp=flame_outputs['landmarks_mp'])
    uv_grid = renderer_output['uv_grid']  # [N, H, W, 2]
    visibility_mask = renderer_output['visibility_mask']  # [N, H, W, 1]
    return uv_grid, visibility_mask

class UVGenerator(object):
    def __init__(self, smirk_base_path: str, smirk_ckpts_path: str, mp_ldmk_model_path: str, device: torch.device):
        """
        Initialize the UVGenerator with the provided SMIRK checkpoint path, SMIRK base path, and device.

        Args:
            smirk_ckpts_path (str): Path to the SMIRK checkpoint file.
            smirk_base_path (str): Base path for the SMIRK model files.
            device (torch.device): The device to load the model onto (e.g., 'cuda' or 'cpu').
        """
        self.device = device
        self.img_sz = 224
        self.smirk_encoder = init_smirk(smirk_ckpts_path, device)
        self.flame = init_flame(smirk_base_path, device)
        self.renderer = init_renderer(smirk_base_path, device)
        self.ldmk_detector = init_mp_lmk_detector(mp_ldmk_model_path)
        
    def forward(self, imgs: torch.Tensor, align_ldmks: bool = False, batch_size: int = -1) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Generate UV coordinates for the input images.

        Args:
            imgs (torch.Tensor): Input images in the shape of (B, 3, H, H), RGB, range [0, 1], dtype torch.float32.
            align_ldmks (bool): Whether to align the landmarks using similarity transform. Default is False. This option is slower but closer to the original SMIRK paper.
            batch_size (int): The batch size for processing images. Default is -1, which means all images are processed in one batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[int]]: A tuple containing:
                - uv_grid (torch.Tensor): UV coordinates in the shape of (B, H, W, 2), where H and W are the height and width of the rendered image.
                - visibility_mask (torch.Tensor): Visibility mask in the shape of (B, 1, H, W), indicating which pixels are visible.
                - index (List[int]): List of indices of images where no face was detected by mediapipe face landmarker(only useful if align_ldmks is True).
                    for these images, the UV maps are generated without alignment and may thus be incorrect if the images are too extreme.
        """
        if batch_size <= 0:
            return self.run(imgs, align_ldmks)
        else:
            total_num = imgs.shape[0]
            uvs, bin_masks, noface_idxs = [], [], []
            for start_idx in range(0, total_num, batch_size):
                end_idx = min(start_idx + batch_size, total_num)
                _uvs, _bin_masks, _noface_idxs = self.run(imgs[start_idx:end_idx], align_ldmks)
                uvs.append(_uvs.cpu())
                bin_masks.append(_bin_masks.cpu())
                noface_idxs += [idx + start_idx for idx in _noface_idxs]
            uvs = torch.cat(uvs, dim=0).to(self.device)
            bin_masks = torch.cat(bin_masks, dim=0).to(self.device)
            return uvs, bin_masks, noface_idxs

    def run(self, imgs: torch.Tensor, align_ldmks: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Generate UV coordinates for the input images.

        Args:
            imgs (torch.Tensor): Input images in the shape of (B, 3, H, H), RGB, range [0, 1], dtype torch.float32.
            align_ldmks (bool): Whether to align the landmarks using similarity transform. Default is False. This option is slower but closer to the original SMIRK paper.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[int]]: A tuple containing:
                - uv_grid (torch.Tensor): UV coordinates in the shape of (B, H, W, 2), where H and W are the height and width of the rendered image.
                - visibility_mask (torch.Tensor): Visibility mask in the shape of (B, 1, H, W), indicating which pixels are visible.
                - index (List[int]): List of indices of images where no face was detected by mediapipe face landmarker(only useful if align_ldmks is True).
                    for these images, the UV maps are generated without alignment and may thus be incorrect if the images are too extreme.
        """
        _, _, H, _ = imgs.shape
        if H != self.img_sz:
            resz_imgs = F.interpolate(imgs, size=(self.img_sz, self.img_sz), mode='bilinear', align_corners=False)
        else:
            resz_imgs = imgs.clone()
        if not align_ldmks:
            uvs, bin_mask = gen_uvs(resz_imgs, self.smirk_encoder, self.flame, self.renderer, self.device)
            return uvs, bin_mask.permute(0, 3, 1, 2), []
        else:
            uvs, bin_masks = [], []
            noface_idxs = []
            imgs_np = resz_imgs.detach().mul(255.).permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)  # [N, H, W, 3], uint8, RGB
            for img_idx, img_np in enumerate(imgs_np):
                ldmks = run_mediapipe(img_np, self.ldmk_detector)  # [478, 3], float32
                if ldmks is None:
                    noface_idxs.append(img_idx)
                    uv, bin_mask = gen_uvs(resz_imgs[[img_idx]], self.smirk_encoder, self.flame, self.renderer, self.device)
                    uvs.append(uv[0].detach().cpu())
                    bin_masks.append(bin_mask[0].detach().cpu())
                    continue
                ldmks = ldmks[:, :2]
                tform = estimate_tform(ldmks, self.img_sz, self.img_sz, scale=1.4, image_size=self.img_sz)
                aligned_img = warp(img_np, tform.inverse, output_shape=(self.img_sz, self.img_sz), preserve_range=True).astype(np.uint8)
                aligned_img_tensor = torch.tensor(aligned_img, dtype=torch.float32).to(self.device).permute(2, 0, 1).unsqueeze(0) / 255.  # [1, 3, H, W], float32, RGB, [0, 1]
                uv, bin_mask = gen_uvs(aligned_img_tensor, self.smirk_encoder, self.flame, self.renderer, self.device)  # [1, H, W, 2], [1, H, W, 1]
                uv_np = uv.detach().cpu().numpy()
                bin_mask_np = bin_mask.detach().cpu().numpy()
                uv = torch.tensor(restore(uv_np, np.expand_dims(tform.params, axis=0), self.img_sz), dtype=torch.float32)
                bin_mask = torch.tensor(restore(bin_mask_np, np.expand_dims(tform.params, axis=0), self.img_sz), dtype=torch.float32)
                uvs.append(uv[0])
                bin_masks.append(bin_mask[0])
            uvs = torch.stack(uvs, dim=0).to(self.device)  # [N, H, W, 2]
            bin_masks = torch.stack(bin_masks, dim=0).to(self.device)  # [N, H, W, 1]
            return uvs, bin_masks.permute(0, 3, 1, 2), noface_idxs
