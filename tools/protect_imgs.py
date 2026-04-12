import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
import datetime

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
import torch.nn.functional as F
import cv2
from PIL import Image

from protego.FaceDetection import FD
from protego.utils import load_mask, crop_face
from protego import BASE_PATH
from protego.UVMapping import UVGenerator

if __name__ == "__main__":
    with torch.no_grad():
        ####################################################################################################################
        # Configuration
        ####################################################################################################################
        device = torch.device('cuda:7')
        protectee_name = "Bradley_Cooper"
        src_img_folder = f"{BASE_PATH}/face_db/face_scrub/{protectee_name}"
        dst_img_folder = f"{BASE_PATH}/results/imgs/{protectee_name}_protected"
        mask_names = ['default', 'univ_mask.npy']
        need_detection = False
        three_d = True
        use_bin_mask = True
        epsilon = 16 / 255.
        ####################################################################################################################
        # Config paths
        mask_path = os.path.join(BASE_PATH, 'experiments', mask_names[0], protectee_name, mask_names[1])
        os.makedirs(dst_img_folder, exist_ok=True)
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')

        # Init models
        fd = FD(model_name='mtcnn', device=device) if need_detection else None
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        mask = load_mask(mask_path, device=device)

        # Main loop
        frame_cnt = 0
        total_proc_time = 0.
        img_paths = [os.path.join(src_img_folder, fname) for fname in os.listdir(src_img_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            dst_img_path = os.path.join(dst_img_folder, img_name)
            frame = cv2.imread(img_path)
            frame_cnt += 1
            start_time = datetime.datetime.now()
            if need_detection:
                frame_pt = torch.tensor(frame, dtype=torch.float32, device=device).permute(2, 0, 1).div(255.)[[2, 1, 0], :, :]  # [1, 3, H, W], RGB, 0-1
                cropped_face, face_pos = crop_face(img=frame_pt, detector=fd, crop_loosen=1.3, verbose=False, strict=False)
                if cropped_face is None or face_pos is None:
                    print(f"Frame {frame_cnt}: No face detected, skip protection.")
                    continue
                face_pos = list(face_pos)
                orig_face_h, orig_face_w = cropped_face.shape[1], cropped_face.shape[2]
                resized_face = F.interpolate(cropped_face.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False) # [1, 3, 224, 224], 0-1
                uv, bin_mask, _ = uvmapper.forward(imgs=resized_face, align_ldmks=False, batch_size=-1)
                if three_d:
                    pert = F.grid_sample(mask, uv, align_corners=True, mode='bilinear')  # [1, 3, 224, 224]
                else:
                    pert = mask
                if use_bin_mask:
                    pert *= bin_mask
                pert = F.interpolate(pert, size=(orig_face_h, orig_face_w), mode='bilinear', align_corners=False).clamp_(-epsilon, epsilon)
                protected_face: torch.Tensor = (cropped_face + pert.squeeze(0)).clamp_(0., 1.)  # [3, H, W], 0-1
                frame_pt[:, face_pos[1]:face_pos[3], face_pos[0]:face_pos[2]] = protected_face.contiguous()
                protected_frame = frame_pt.permute(1, 2, 0).mul(255.).to(torch.uint8)[:, :, [2, 1, 0]].cpu().contiguous().numpy()
            else:
                frame_pt = torch.tensor(frame, dtype=torch.float32, device=device).permute(2, 0, 1).div(255.)[[2, 1, 0], :, :].unsqueeze(0)  # [1, 3, H, W], RGB, 0-1
                frame_pt = F.interpolate(frame_pt, size=(224, 224), mode='bilinear', align_corners=False) # [1, 3, 224, 224], 0-1
                uv, bin_mask, _ = uvmapper.forward(imgs=frame_pt, align_ldmks=False, batch_size=-1)
                if three_d:
                    pert = F.grid_sample(mask, uv, align_corners=True, mode='bilinear')  # [1, 3, 224, 224]
                else:
                    pert = mask
                if use_bin_mask:
                    pert *= bin_mask
                pert.clamp_(-epsilon, epsilon)
                protected_frame: torch.Tensor = (frame_pt + pert).clamp_(0., 1.)  # [1, 3, H, W], 0-1
                protected_frame = protected_frame.squeeze(0).permute(1, 2, 0).mul(255.).to(torch.uint8)[:, :, [2, 1, 0]].cpu().contiguous().numpy()
            end_time = datetime.datetime.now()
            proc_time = (end_time - start_time).total_seconds()
            total_proc_time += proc_time
            cv2.imwrite(dst_img_path, protected_frame)
        print(f"Processed {frame_cnt} images, average time per image: {total_proc_time / frame_cnt:.4f} seconds.")