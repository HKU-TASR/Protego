import os

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
import tqdm

from protego import BASE_PATH
from protego.utils import preextract_features
from protego.FacialRecognition import FR, BASIC_POOL, SPECIAL_POOL, FINETUNE_POOL

if __name__ == "__main__":
    ######################### Configuration #########################
    device = torch.device('cuda:0')
    mode = 'original' # 'original', 'compressed', 'both'
    face_db_base_paths = [f"{BASE_PATH}/face_db/bfw_gender_woman_cropped", 
                          f"{BASE_PATH}/face_db/bfw_gender_man_cropped"]
    fr_names = ['ir50_adaface_casia']
    compression_methods = ['none', 'gaussian', 'median', 'jpeg', 'resize', 'quantize', 'vid_codec']
    compression_cfgs = {
        'none': {}, 
        # Gaussian Filter
        'gaussian': {
            'kernel_size': 9, 
            'sigma': 2.0,
        }, 
        # Median Filter
        'median': {
            'kernel_size': 9
        }, 
        # JPEG Compression
        'jpeg': {
            'quality': 70, 
        }, 
        # Resize
        'resize': {
            'resz_percentage': 0.4,
            'mode': 'bicubic'
        }, 
        'quantize': {
            'precision': 'uint8'
        }, 
        'vid_codec': {
            'codec': 'h264',
            'crf': 32,
            'preset': 'faster'
        }
    }
    #################################################################
    with torch.no_grad():
        for fr_idx, fr_name in enumerate(fr_names):
            print(f"Processing FR model {fr_idx+1}/{len(fr_names)}: {fr_name}")
            fr = FR(model_name=fr_name, device=device)
            for db_path in face_db_base_paths:
                pbar = tqdm.tqdm([name for name in os.listdir(db_path) if not name.startswith(('.', '_'))], desc=f"Processing face DB at {db_path}")
                for name in pbar: 
                    personal_path = os.path.join(db_path, name)
                    if mode in ['original', 'both']:
                        preextract_features(base_path=personal_path, fr=fr, device=device, save_name=f"{fr.model_name}.pt")
                    if mode in ['compressed', 'both']:
                        for method in compression_methods:
                            cfgs_str = ''
                            for k, v in compression_cfgs[method].items():
                                cfgs_str += f"_{k}_{v}"
                            f_name = f"{fr_name}_{method}{cfgs_str}.pt"
                            preextract_features(base_path=personal_path, fr=fr, device=device, save_name=f_name, compression_cfg=(method, compression_cfgs[method]))

