import os

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
import tqdm

from protego import BASE_PATH
from protego.utils import preextract_features
from protego.FacialRecognition import FR

if __name__ == "__main__":
    ######################### Configuration #########################
    device = torch.device('cuda:0')
    # Folders whose sub-directories (one per identity) will be turned into
    # pre-extracted feature databases for retrieval evaluation.
    face_db_base_paths = [f"{BASE_PATH}/face_db/face_scrub/_noise_db"]
    fr_names = ['ir50_adaface_casia']
    #################################################################
    with torch.no_grad():
        for fr_idx, fr_name in enumerate(fr_names):
            print(f"Processing FR model {fr_idx+1}/{len(fr_names)}: {fr_name}")
            fr = FR(model_name=fr_name, device=device)
            for db_path in face_db_base_paths:
                pbar = tqdm.tqdm([name for name in os.listdir(db_path) if not name.startswith(('.', '_'))], desc=f"Processing face DB at {db_path}")
                for name in pbar: 
                    personal_path = os.path.join(db_path, name)
                    preextract_features(base_path=personal_path, fr=fr, device=device, save_name=f"{fr.model_name}.pt")

