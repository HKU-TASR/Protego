import math
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torchvision.ops import nms, roi_align
import numpy as np
import tqdm

from .nets import PNet, RNet, ONet

class MTCNN(object):
    @torch.no_grad()
    def __init__(
        self,
        weight_paths: Dict[str, str] = {'pnet': "weights/pnet.npy", 'rnet': "weights/rnet.npy", 'onet': "weights/onet.npy"},
        device: torch.device = None,
    ) -> None:
        """
        Init a MTCNN model.

        Args:
            weight_paths: Dictionary mapping network names to their weight file paths.
            device: Device to run the model on
        """
        self.device = device or torch.device('cpu')

        self.pnet = PNet(weight_path=weight_paths['pnet']).to(self.device).eval()
        self.rnet = RNet(weight_path=weight_paths['rnet']).to(self.device).eval()
        self.onet = ONet(weight_path=weight_paths['onet']).to(self.device).eval()

        self._min_det_size = 12.0

    @torch.inference_mode()
    def __call__(self, imgs: Union[torch.Tensor, List[torch.Tensor]]) -> Optional[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
        """
        Alias for the detect method.
        """
        return self.detect(imgs)

    @torch.inference_mode()
    def detect(self, imgs: Union[torch.Tensor, List[torch.Tensor]], verbose: bool = False, return_zero: bool = True, return_type: str = 'any', 
                min_face_size: float = 20.0, scale_factor: float = math.sqrt(0.5), conf_threshes: List[float] = [0.6, 0.7, 0.8], nms_threshes: List[float] = [0.7, 0.7, 0.7]
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]], Tuple[None, None]]:
        """
        Args:
            imgs: FloatTensor [B,3,H,W] or [3,H,W], RGB, Range [0,255].
                or List[FloatTensor] of shape [3,H,W], [B,3,H,W] or a mix of both.
            verbose: If True, enables verbose logging.
            return_zero: If False, returns (None, None) when no faces are detected. Otherwise, return empty tensors with B = 0, N = 0, C = 5 for boxes and 10 for landmarks.
            return_type: Type of the return value. Can be 'any', 'tensor', or 'list'. If 'any', the return type is the same as the input. 
            conf_threshes: List of confidence thresholds for PNet, RNet, and ONet.
            nms_threshes: List of NMS thresholds for PNet, RNet, and ONet.
            min_face_size: Minimum size of faces to detect.
            scale_factor: Scale factor for image pyramid.

        Returns:
            boxes: Tensor [B,N,5]  (x1,y1,x2,y2,score) or List[Tensor] of shape [N,5] or None or empty tensor of shape [0,0,5]
            landmarks: Tensor [B,N,10] or List[Tensor] of shape [N,10] or None or empty tensor of shape [0,0,10]
            The sequence of the 5 landmarks are: 
            [x1, y1, x2, y2, ..., x5, y5] coordinates for:
            left eye, right eye, nose, left mouth corner, right mouth corner
        """
        assert isinstance(imgs, (torch.Tensor, list)), "Input must be a Tensor or a list of Tensors."
        _return_type = 'tensor'
        if isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4:  # [B, 3, H, W]
            imgs = [imgs[[i]] for i in range(imgs.shape[0])]
        elif isinstance(imgs, torch.Tensor) and len(imgs.shape) == 3:  # [3, H, W]
            imgs = [imgs.unsqueeze(0)]
        else:
            _return_type = 'list'
            imgs_list = []
            for img in imgs:
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                    imgs_list.append(img)
                elif len(img.shape) == 4 and img.shape[0] == 1:
                    imgs_list.append(img)
                elif len(img.shape) == 4 and img.shape[0] > 1:
                    imgs_list.extend([img[[i]] for i in range(img.shape[0])])
            imgs = imgs_list
        return_type = return_type if return_type != 'any' else _return_type
        pbar = tqdm.tqdm(enumerate(imgs), desc="Detecting Faces", disable=not verbose)
        bboxes, ldmks = [], []
        for img_idx, img in pbar:
            img: torch.Tensor = img.to(self.device) # [1, 3, H, W]
            #####################################
            # P-Net
            #####################################
            pyramid = self._build_pyramid(img, min_face_size, scale_factor)
            all_bboxes_stage1: List[torch.Tensor] = []
            for pyramid_idx, (scaled_img, scale) in enumerate(pyramid):
                offset, prob = self.pnet(scaled_img) # Offset: [1, 4, H', W'], Probs: [1, 2, H', W']
                prob = prob[0, 1, :, :] # Extract face confidence scores, Probs: [H', W']
                offset = offset[0] # Extract offsets, Offset: [4, H', W']
                stage1_bboxes = self._gen_bboxes(prob, offset, scale, conf_threshes[0])
                if stage1_bboxes is not None:
                    all_bboxes_stage1.append(stage1_bboxes)
            if not all_bboxes_stage1:
                print(f"Image {img_idx} has no faces detected in PNet.")
                continue
            all_bboxes_stage1 = torch.cat(all_bboxes_stage1, dim=0)  # [N, 5] (x1, y1, x2, y2, score)
            keep = nms(all_bboxes_stage1[:, :4], all_bboxes_stage1[:, 4], nms_threshes[0])
            if keep.numel() == 0:
                print(f"Image {img_idx} has no faces detected after PNet and NMS.")
                continue
            all_bboxes_stage1 = all_bboxes_stage1[keep]
            all_bboxes_stage1[:, :4] = self._square(all_bboxes_stage1[:, :4])  # Square the boxes
            all_bboxes_stage1[:, :4] = torch.round(all_bboxes_stage1[:, :4])  # Round the coordinates

            #####################################
            # R-Net
            #####################################
            # Add batch indices for roi_align (requires [N, 5] format: batch_idx, x1, y1, x2, y2)
            batch_indices = torch.zeros(all_bboxes_stage1.shape[0], 1, device=all_bboxes_stage1.device)
            rois_stage1 = torch.cat([batch_indices, all_bboxes_stage1[:, :4]], dim=1)
            crops = roi_align(img, rois_stage1, output_size=(24, 24), spatial_scale=1.0, aligned=True)
            crops = self._rescale_imgs(crops)  # Rescale to [-1, 1]
            offsets, probs = self.rnet(crops)  # Offset: [N, 4], Probs: [N, 2]
            probs = probs[:, 1]  # Extract face confidence scores, Probs: [N]
            keep = probs > conf_threshes[1]
            if not keep.any():
                print(f"Image {img_idx} has no faces detected in RNet.")
                continue
            all_bboxes_stage2 = all_bboxes_stage1[keep]
            all_bboxes_stage2[:, 4] = probs[keep]
            offsets = offsets[keep]  # [N, 4]
            keep = nms(all_bboxes_stage2[:, :4], all_bboxes_stage2[:, 4], nms_threshes[1])
            if keep.numel() == 0:
                print(f"Image {img_idx} has no faces detected after RNet and NMS.")
                continue
            all_bboxes_stage2 = all_bboxes_stage2[keep]
            offsets = offsets[keep]
            all_bboxes_stage2 = self._calibrate(all_bboxes_stage2, offsets)
            all_bboxes_stage2[:, :4] = self._square(all_bboxes_stage2[:, :4])
            all_bboxes_stage2[:, :4] = torch.round(all_bboxes_stage2[:, :4])  # Round the coordinates

            #####################################
            # O-Net
            #####################################
            # Add batch indices for roi_align (requires [N, 5] format: batch_idx, x1, y1, x2, y2)
            batch_indices = torch.zeros(all_bboxes_stage2.shape[0], 1, device=all_bboxes_stage2.device)
            rois_stage2 = torch.cat([batch_indices, all_bboxes_stage2[:, :4]], dim=1)
            crops = roi_align(img, rois_stage2, output_size=(48, 48), spatial_scale=1.0, aligned=True)
            crops = self._rescale_imgs(crops)
            landmarks, offsets, probs = self.onet(crops) # Landmarks: [N, 10], Offsets: [N, 4], Probs: [N, 2]
            probs = probs[:, 1]  # Extract face confidence scores, Probs: [N]
            keep = probs > conf_threshes[2]
            if not keep.any():
                print(f"Image {img_idx} has no faces detected in ONet.")
                continue
            all_bboxes_stage3 = all_bboxes_stage2[keep]
            all_bboxes_stage3[:, 4] = probs[keep]
            offsets = offsets[keep]  # [N, 4]
            landmarks = landmarks[keep]  # [N, 10]
            w = all_bboxes_stage3[:, 2] - all_bboxes_stage3[:, 0] + 1.0
            h = all_bboxes_stage3[:, 3] - all_bboxes_stage3[:, 1] + 1.0
            xmin, ymin = all_bboxes_stage3[:, 0], all_bboxes_stage3[:, 1]
            landmarks[:, 0:5] = xmin.unsqueeze(1) + w.unsqueeze(1)*landmarks[:, 0:5]
            landmarks[:, 5:10] = ymin.unsqueeze(1) + h.unsqueeze(1)*landmarks[:, 5:10]
            all_bboxes_stage3 = self._calibrate(all_bboxes_stage3, offsets)
            keep = nms(all_bboxes_stage3[:, :4], all_bboxes_stage3[:, 4], nms_threshes[2])
            if keep.numel() == 0:
                print(f"Image {img_idx} has no faces detected after ONet and NMS.")
                continue
            all_bboxes_stage3 = all_bboxes_stage3[keep]
            landmarks = landmarks[keep]  # [N, 10]
            bboxes.append(all_bboxes_stage3.unsqueeze(0).cpu())
            ldmks.append(landmarks.unsqueeze(0).cpu())

        if len(bboxes) == 0 or len(ldmks) == 0:
            return (None, None) if not return_zero else ((torch.empty(0, 0, 5), torch.empty(0, 0, 10)) if return_type == 'tensor' else ([], []))
        if return_type == 'tensor':
            return torch.cat(bboxes, dim=0), torch.cat(ldmks, dim=0)
        if return_type == 'list':
            return [b[0] for b in bboxes], [l[0] for l in ldmks]

    # ----------------- Helper Functions -----------------
    def _rescale_imgs(self, img: torch.Tensor) -> torch.Tensor:
        """
        Rescales images from [0, 255] to [-1, 1].

        Args:
            img: [B, 3, H, W]

        Returns:
            Rescaled images: [B, 3, H, W]
        """
        return (img - 127.5) * 0.0078125

    def _build_pyramid(self, img: torch.Tensor, min_face_size: float, scale_factor: float) -> List[Tuple[torch.Tensor, float]]:
        """
        Args:
            img: [B, 3, H, W]

        Returns:
            List of (scaled_imgs, scale) tuples
        """
        _, _, h, w = img.shape
        min_detection_sz = self._min_det_size
        min_size = min(h, w)
        
        # Calculate the minimum scale such that the minimum side becomes min_face_size
        min_scale = min_face_size / min_size

        pyramid = []
        scale = 1.0
        
        while scale >= min_scale:
            scaled_imgs = F.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)
            preprocessed_imgs = self._rescale_imgs(scaled_imgs)
            pyramid.append((preprocessed_imgs, scale))
            scale *= scale_factor
            
        return pyramid
    
    def _gen_bboxes(self, prob: torch.Tensor, offset: torch.Tensor, scale: float, thresh: float) -> Optional[torch.Tensor]:
        """
        Args:
            probs: [H, W]
            offsets: [4, H, W]
            scale: the current scale
            thresh: the confidence threshold

        Returns:
            boxes: [N, 5] (x1, y1, x2, y2, score) or None (if no boxes are found to have confidence above the threshold)
        """
        stride = 2.
        cell_sz = 12.
        mask = prob > thresh
        if not mask.any():
            return None
        inds = mask.nonzero(as_tuple=False)  # [K,2] (y,x)
        scores = prob[mask]  # [K]
        oy = inds[:, 0]
        ox = inds[:, 1]
        x1 = (stride * ox + 1.0) / scale
        y1 = (stride * oy + 1.0) / scale
        x2 = (stride * ox + 1.0 + cell_sz) / scale
        y2 = (stride * oy + 1.0 + cell_sz) / scale

        tx1 = offset[0, oy, ox]
        ty1 = offset[1, oy, ox]
        tx2 = offset[2, oy, ox]
        ty2 = offset[3, oy, ox]

        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        x1 = x1 + tx1 * w
        y1 = y1 + ty1 * h
        x2 = x2 + tx2 * w
        y2 = y2 + ty2 * h

        boxes = torch.stack([x1, y1, x2, y2, scores], dim=1)
        return boxes
    
    def _square(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Square the bounding boxes.

        Args:
            bboxes: [N, K≥4] (x1, y1, x2, y2, ...)

        Returns:
            Squared bounding boxes: [N, K]
        """
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        max_side = torch.maximum(w, h)
        nx1 = x1 + w * 0.5 - max_side * 0.5
        ny1 = y1 + h * 0.5 - max_side * 0.5
        nx2 = nx1 + max_side - 1.0
        ny2 = ny1 + max_side - 1.0
        bboxes[:, 0:4] = torch.stack([nx1, ny1, nx2, ny2], dim=1)
        return bboxes
    
    def _calibrate(self, bboxes: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Calibrates the bounding boxes using the offsets.

        Args:
            bboxes: [N, K≥4] (x1, y1, x2, y2, ...)
            offsets: [N, 4] (dx1, dy1, dx2, dy2)

        Returns:
            Calibrated bounding boxes: [N, K]
        """
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        x1 = x1 + offsets[:, 0] * w
        y1 = y1 + offsets[:, 1] * h
        x2 = x2 + offsets[:, 2] * w
        y2 = y2 + offsets[:, 3] * h
        bboxes[:, 0:4] = torch.stack([x1, y1, x2, y2], dim=1)
        return bboxes
