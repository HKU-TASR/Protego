import os
from typing import Union, Dict, Tuple, List
import pickle

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
import torch.nn.functional as F
import yaml

from protego.FacialRecognition import FR
from protego.utils import retrieve, build_facedb, get_usable_img_paths, load_imgs, load_mask
from protego import BASE_PATH
from protego.UVMapping import UVGenerator   

def build_complete_face_db(noise_db_path: str, protectee_path: str, mask_path: str, epsilon: float, uvmapper: UVGenerator, fr: FR, device: torch.device) -> Tuple[Dict[str, Tuple[torch.Tensor, List[str]]], Dict[str, Tuple[torch.Tensor, torch.Tensor, List[str]]]]:
    face_db = build_facedb(db_path=noise_db_path, fr=fr, device=torch.device('cpu'), return_img_paths=True)
    protectee_db = {}
    protectees = sorted([name for name in os.listdir(protectee_path) if os.path.isdir(os.path.join(protectee_path, name)) and not name.startswith(('.', '_'))])
    for protectee in protectees:
        protectee_dir = os.path.join(protectee_path, protectee)
        protectee_mask_dir = os.path.join(mask_path, protectee, 'univ_mask.npy')
        img_paths = get_usable_img_paths(protectee_dir)
        imgs, img_paths = load_imgs(base_dir=None, img_paths=img_paths, img_sz=224, usage_portion=1., drange=1, device=device, return_img_paths=True)
        train_num = int(len(imgs)*0.6)
        imgs = imgs[train_num:]
        img_paths = img_paths[train_num:]
        mask = load_mask(protectee_mask_dir, device=device)
        uvs, bin_masks, _ = uvmapper.forward(imgs=imgs, align_ldmks=False, batch_size=-1)
        perts = F.grid_sample(mask.repeat(uvs.shape[0], 1, 1, 1), uvs, align_corners=True, mode='bilinear').clamp_(-epsilon, epsilon).mul_(bin_masks)
        protected_imgs = torch.clamp(imgs + perts, 0., 1.)
        orig_features = fr(imgs)
        prot_features = fr(protected_imgs)
        protectee_db[protectee] = (orig_features.cpu(), prot_features.cpu(), img_paths)
    return face_db, protectee_db

def get_retrievals(protectee: str, noise_db: Dict[str, Tuple[torch.Tensor, List[str]]], protectee_db: Dict[str, Tuple[torch.Tensor, torch.Tensor, List[str]]], device: torch.device) -> Dict[str, Dict[Tuple[str, str], List[Tuple[str, str]]]]:
    for other_protectee_name, (orig_feats, prot_feats, img_paths) in protectee_db.items():
        if other_protectee_name == protectee:
            continue
        prot_feature_num = int(orig_feats.shape[0] * 0.5)
        # Create a copy to avoid modifying the original list
        modified_img_paths = img_paths.copy()
        for i in range(prot_feature_num):
            modified_img_paths[i] += "(protected)"
        noise_db[other_protectee_name] = (torch.cat((prot_feats[:prot_feature_num], orig_feats[prot_feature_num:])), modified_img_paths)
    db_features, db_labels, db_img_paths = [], [], []
    for name, (features, img_paths) in noise_db.items():
        db_features.append(features)
        db_labels.extend([name]*features.shape[0])
        db_img_paths.extend(img_paths)
    db_features = torch.cat(db_features).to(device)

    res = {}
    # Use protectee's protected features to retrieve protectee's original features in DB
    protectee_orig_feats, protectee_prot_feats, protectee_img_paths = protectee_db[protectee]
    query_feature_num = int(protectee_orig_feats.shape[0] * 0.5)
    protectee_db_features_num = protectee_orig_feats.shape[0] - query_feature_num

    query_features = protectee_prot_feats[:query_feature_num].to(device)
    query_img_paths = [path + "(protected)" for path in protectee_img_paths[:query_feature_num]]
    query_labels = [protectee]*query_feature_num

    protectee_db_features = protectee_orig_feats[query_feature_num:].to(device)
    protectee_db_img_paths = protectee_img_paths[query_feature_num:]
    protectee_db_labels = [protectee]*protectee_db_features_num

    db_features_extended = torch.cat((db_features, protectee_db_features), dim=0)
    db_labels_extended = db_labels + protectee_db_labels
    img_paths_extended = db_img_paths + protectee_db_img_paths

    _, retrieved_idxs = retrieve(db=db_features_extended, db_labels=db_labels_extended, 
                                 queries=query_features, query_labels=query_labels, 
                                 dist_func='cosine', topk=protectee_db_features_num, 
                                 sorted_retrieval=True, return_retrieved_idxs=True)
    retrievals = {}
    for q_idx, retrieved_idx_list in enumerate(retrieved_idxs):
        query_img_path = query_img_paths[q_idx]
        retrievals[(protectee, query_img_path)] = []
        for ridx in retrieved_idx_list:
            retrieved_label = db_labels_extended[ridx]
            retrieved_img_path = img_paths_extended[ridx]
            retrievals[(protectee, query_img_path)].append((retrieved_label, retrieved_img_path))
    res['2b'] = retrievals

    # Use protectee's protected features to retrieve protectee's protected features in DB
    protectee_db_features = protectee_prot_feats[query_feature_num:].to(device)
    protectee_db_img_paths = [path + "(protected)" for path in protectee_img_paths[query_feature_num:]]
    protectee_db_labels = [protectee]*protectee_db_features_num

    db_features_extended = torch.cat((db_features, protectee_db_features), dim=0)
    db_labels_extended = db_labels + protectee_db_labels
    img_paths_extended = db_img_paths + protectee_db_img_paths

    _, retrieved_idxs = retrieve(db=db_features_extended, db_labels=db_labels_extended, 
                                 queries=query_features, query_labels=query_labels,
                                 dist_func='cosine', topk=protectee_db_features_num,
                                 sorted_retrieval=True, return_retrieved_idxs=True)
    retrievals = {}
    for q_idx, retrieved_idx_list in enumerate(retrieved_idxs):
        query_img_path = query_img_paths[q_idx]
        retrievals[(protectee, query_img_path)] = []
        for ridx in retrieved_idx_list:
            retrieved_label = db_labels_extended[ridx]
            retrieved_img_path = img_paths_extended[ridx]
            retrievals[(protectee, query_img_path)].append((retrieved_label, retrieved_img_path))
    res['2a'] = retrievals

    return res

if __name__ == "__main__":
    with torch.no_grad():
        device = torch.device("cuda:6")
        noise_db_path = f"{BASE_PATH}/face_db/face_scrub/_noise_db"
        protectee_path = f"{BASE_PATH}/face_db/face_scrub"
        mask_path = f"{BASE_PATH}/experiments/default"
        epsilon = 16 / 255.
        fr_name = 'ir50_adaface_casia'
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        fr = FR(model_name=fr_name, device=device)
        
        complete_res = {}
        protectees = sorted([name for name in os.listdir(protectee_path) if os.path.isdir(os.path.join(protectee_path, name)) and not name.startswith(('.', '_'))])
        for protectee in protectees:
            noise_db, protectee_db = build_complete_face_db(noise_db_path=noise_db_path, protectee_path=protectee_path, mask_path=mask_path, epsilon=epsilon, uvmapper=uvmapper, fr=fr, device=device)
            retrievals = get_retrievals(protectee=protectee, noise_db=noise_db, protectee_db=protectee_db, device=device)
            complete_res.update(retrievals)
            save_path = f"/home/zlwang/ProtegoPlus/trash/retrievals/face_scrub_{fr_name}_{protectee}_epsilon{int(epsilon*255)}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(complete_res, f)
            # Convert tuple keys to strings for YAML compatibility
            yaml_res = {}
            for case_key, retrievals in complete_res.items():
                yaml_res[case_key] = {}
                for (protectee, query_path), retrieved_list in retrievals.items():
                    yaml_res[case_key][f"{protectee}|{query_path}"] = [f"{label}|{path}" for label, path in retrieved_list]
            save_path = f"/home/zlwang/ProtegoPlus/trash/retrievals/face_scrub_{fr_name}_{protectee}_epsilon{int(epsilon*255)}.yaml"
            with open(save_path, 'w') as f:
                yaml.dump(yaml_res, f)