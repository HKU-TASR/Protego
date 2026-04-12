import os
from typing import List, Dict, Any, Optional
import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import yaml
from omegaconf import OmegaConf
import tqdm

from .FacialRecognition import FR
from .FaceDetection import FD
from .UVMapping import UVGenerator
from .utils import load_imgs, load_mask, build_facedb, build_compressed_face_db, crop_face, complete_del, visualize_mask, eval_masks, compression_eval, eval_mask_end2end, get_augmented_ims
from .compression import compress
from . import BASE_PATH

def run(cfgs: OmegaConf, mode: str, data: Dict[str, Dict[str, List[str]]], train: Optional[Any] = None) -> None:
    """
    Run training or evaluation based on the provided configuration.

    Args:
        cfgs (OmegaConf): Configuration object containing settings for the run.
        mode (str): Mode of operation, 'train' or 'eval'. 
        data (Dict[str, Dict[str, List[str]]]): Dictionary containing the paths to each image.
            {"Bradley_Cooper": {
                'train': [list of training image paths],
                'eval': [list of evaluation image paths]
            }, ...}
        train (callable, optional): Training function to be called if mode is 'train'.
    """
    assert mode in ['train', 'eval'], "Mode must be either 'train' or 'eval'."
    ####################################################################################################################
    # Set the paths
    ####################################################################################################################
    device = torch.device(cfgs.device)
    face_db_path = os.path.join(BASE_PATH, "face_db", cfgs.eval_db)
    noise_db_path = os.path.join(face_db_path, '_noise_db')
    exp_name = cfgs.exp_name
    res_base_path = os.path.join(BASE_PATH, 'results', 'eval') if mode == 'eval' else os.path.join(BASE_PATH, 'experiments')
    res_path = os.path.join(res_base_path, exp_name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        # print(f"The experiment results will be saved in {res_path}.")
    else:
        new_exp_name = f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        cfgs.exp_name = new_exp_name
        # print(f"Experiment {exp_name} already exists. Changing it to {new_exp_name}")
        res_path = os.path.join(res_base_path, new_exp_name)
        os.makedirs(res_path)
    if mode == 'eval':
        mask_base_path = os.path.join(BASE_PATH, 'experiments', cfgs.mask_name[0])

    with torch.no_grad():
        ####################################################################################################################
        # Init SMIRK
        ####################################################################################################################
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        ####################################################################################################################
        # Init FD if needed
        ####################################################################################################################
        if cfgs.need_cropping or cfgs.end2end_eval:
            fd = FD(model_name=cfgs.fd_name, device=device)
    ####################################################################################################################
    # Init FRs
    ####################################################################################################################
    train_frs = [FR(model_name=fr_name, device=device) for fr_name in cfgs.train_fr_names]
    eval_frs = [FR(model_name=fr_name, device=device) for fr_name in cfgs.eval_fr_names]
    ####################################################################################################################
    # Main loop
    ####################################################################################################################
    if cfgs.eval_scene == 1:
        protectee_masks, protectee_eval_data, protectee_no_face_imgs = {}, {}, {}
    training_times = []
    for protectee_idx, (protectee, protectee_data) in enumerate(data.items()):
        ####################################################################################################################
        # Enumerate through all the protectees
        ####################################################################################################################
        if not (cfgs.eval_scene == 1 and mode == 'eval'):
            print("\n"+"#" * 50)
            print(f"Protectee: {protectee} ({protectee_idx+1}/{len(data)})")
            print("#" * 50+ "\n")
        res_save_path = os.path.join(res_path, protectee)
        os.makedirs(res_save_path, exist_ok=False)
        print(f"Created directory {res_save_path} for protectee {protectee}.")
        with open(os.path.join(res_save_path, 'cfgs.yaml'), 'w') as f:
            OmegaConf.save(cfgs, f)
        if mode == 'train':
            with torch.no_grad():
                if not cfgs.need_cropping:
                    training_imgs: torch.Tensor = load_imgs(img_paths=protectee_data['train'], img_sz=cfgs.mask_size, drange=255, device=device)
                    #print(training_imgs.shape, training_imgs.min(), training_imgs.max())
                else:
                    training_imgs: List[torch.Tensor] = load_imgs(img_paths=protectee_data['train'], img_sz=-1, drange=255, device=device)
                    _cropped_imgs = []
                    _no_face = []
                    for img_idx, img in enumerate(training_imgs):
                        cropped_face, pos = crop_face(img=img.squeeze(0), detector=fd, crop_loosen=cfgs.crop_loosen, verbose=True)
                        if cropped_face is None or pos is None:
                            _no_face.append(img_idx)
                            continue
                        _cropped_imgs.append(F.interpolate(cropped_face.unsqueeze(0), size=(cfgs.mask_size, cfgs.mask_size), mode='bilinear', align_corners=True, antialias=True).squeeze(0))
                    print(f"{len(_no_face)} training images do not have detectable faces and are ignored:")
                    for idx in _no_face:
                        print(f"{protectee_data['train'][idx]}")
                    training_imgs: torch.Tensor = torch.stack(_cropped_imgs, dim=0)
                    del _cropped_imgs
                    complete_del()
                #print(training_imgs.shape, training_imgs.min(), training_imgs.max())
                training_imgs.div_(255.)
                #print(training_imgs.shape, training_imgs.min(), training_imgs.max())
                img_num = training_imgs.shape[0]
                # Ensure UV mapping runs on the target device
                train_uvs, train_bin_masks, no_faces = uvmapper.forward(training_imgs,align_ldmks=cfgs.uv_gen_align_ldmk,batch_size=cfgs.uv_gen_batch)
                #print(training_imgs.shape, training_imgs.min(), training_imgs.max())
                if cfgs.uv_gen_align_ldmk:
                    training_imgs = [img for idx, img in enumerate(training_imgs) if idx not in no_faces]
                    train_uvs = [uv for idx, uv in enumerate(train_uvs) if idx not in no_faces]
                    train_bin_masks = [mask for idx, mask in enumerate(train_bin_masks) if idx not in no_faces]
                    training_imgs = torch.stack(training_imgs, dim=0)
                    train_uvs = torch.stack(train_uvs, dim=0)
                    train_bin_masks = torch.stack(train_bin_masks, dim=0)
                training_imgs = training_imgs.cpu()
                train_uvs = train_uvs.cpu()
                train_bin_masks = train_bin_masks.cpu()
                train_dl = DataLoader(
                    dataset=TensorDataset(training_imgs, train_uvs, train_bin_masks),
                    batch_size=cfgs.batch_size,
                    shuffle=cfgs.shuffle,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True
                )
                del training_imgs, train_uvs, train_bin_masks
                complete_del()
            train_start = datetime.datetime.now()
            univ_mask: torch.Tensor = train(cfgs=cfgs, frs=train_frs, train_dl=train_dl, results_save_path=res_save_path)
            print(univ_mask.shape, univ_mask.min(), univ_mask.max())
            training_time = (datetime.datetime.now() - train_start).total_seconds()
            print(f"Training time for protectee {protectee}: {training_time:.2f} seconds.")
            training_times.append(training_time)
            if cfgs.save_univ_mask:
                np.save(os.path.join(res_save_path, "univ_mask.npy"), univ_mask.contiguous().numpy())
            if cfgs.visualize_interval > 0:
                for img_idx in range(0, img_num, cfgs.visualize_interval):
                    visualize_mask(
                        orig_img=train_dl.dataset[img_idx][0].to(device),
                        uv=train_dl.dataset[img_idx][1].to(device),
                        bin_mask=train_dl.dataset[img_idx][2].to(device),
                        univ_mask=univ_mask.clone().to(device),
                        epsilon=cfgs.epsilon,
                        save_path=os.path.join(res_save_path, f"train_vis_{img_idx}.jpg"),
                        use_bin_mask=cfgs.bin_mask,
                        three_d=cfgs.three_d)
            del train_dl
            complete_del()
        elif mode == 'eval':
            mask_path = os.path.join(mask_base_path, protectee, cfgs.mask_name[1])
            assert os.path.exists(mask_path), f"Mask file {mask_path} does not exist."
            univ_mask = load_mask(mask_path=mask_path, device=device)

        if cfgs.eval_scene == 1:
            protectee_masks[protectee] = univ_mask.clone().to(torch.device('cpu'))
        
        with torch.no_grad():
            if cfgs.need_cropping or cfgs.end2end_eval:
                eval_imgs = load_imgs(img_paths=protectee_data['eval'], img_sz=-1, usage_portion=1.0, drange=255, device=device)
                _cropped_imgs = []
                no_face = []
                for img_idx, img in enumerate(eval_imgs):
                    cropped_face, pos = crop_face(img=img.squeeze(0), detector=fd, crop_loosen=cfgs.crop_loosen, verbose=True)
                    if cropped_face is None or pos is None:
                        no_face.append(img_idx)
                        continue
                    _cropped_imgs.append(F.interpolate(cropped_face.unsqueeze(0), size=(cfgs.mask_size, cfgs.mask_size), mode='bilinear', align_corners=True, antialias=True).squeeze(0))
                print(f"{len(no_face)} eval images do not have detectable faces and are ignored:")
                for idx in no_face:
                    print(f"{protectee_data['eval'][idx]}")
                if cfgs.eval_scene == 1:
                    protectee_no_face_imgs[protectee] = no_face
                eval_imgs = torch.stack(_cropped_imgs, dim=0)
                del _cropped_imgs
                complete_del()
            else:
                eval_imgs = load_imgs(img_paths=protectee_data['eval'], img_sz=cfgs.mask_size, usage_portion=1.0, drange=255, device=device)
            eval_imgs.div_(255.)
            img_num = eval_imgs.shape[0]
            # Ensure UV mapping runs on the target device, then move tensors back to CPU for DataLoader workers
            if cfgs.input_robustness_eval:
                eval_imgs = get_augmented_ims(orig_imgs=eval_imgs, methods = cfgs.input_robustness_methods, aug_cfgs=cfgs.input_robustness_cfgs)
            eval_uvs, eval_bin_masks, _ = uvmapper.forward(eval_imgs)
            eval_imgs = eval_imgs.cpu()
            eval_uvs = eval_uvs.cpu()
            eval_bin_masks = eval_bin_masks.cpu()
            if cfgs.visualize_interval > 0:
                for img_idx in range(0, img_num, cfgs.visualize_interval):
                    visualize_mask(
                        orig_img=eval_imgs[img_idx].to(device),
                        uv=eval_uvs[img_idx].to(device),
                        bin_mask=eval_bin_masks[img_idx].to(device),
                        univ_mask=univ_mask.clone().to(device),
                        epsilon=cfgs.epsilon,
                        save_path=os.path.join(res_save_path, f"eval_vis_{img_idx}.jpg"),
                        use_bin_mask=cfgs.bin_mask,
                        three_d=cfgs.three_d)
            if cfgs.eval_scene == 1:
                protectee_eval_data[protectee] = {
                    'imgs': eval_imgs,
                    'uvs': eval_uvs,
                    'bin_masks': eval_bin_masks,
                }
                continue
            eval_dl = DataLoader(
                dataset=TensorDataset(eval_imgs, eval_uvs, eval_bin_masks),
                batch_size=16,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            del eval_imgs, eval_uvs, eval_bin_masks
            complete_del()
            if cfgs.replication_attack:
                mask_path = os.path.join(BASE_PATH, 'experiments', cfgs.replication_mask_name[0], protectee, cfgs.replication_mask_name[1])
                rep_mask = load_mask(mask_path=mask_path, device=device)
            else:
                rep_mask = None
            for eval_fr in eval_frs:
                noise_db = build_facedb(db_path=noise_db_path, fr=eval_fr, device=device)
                res = eval_masks(
                    three_d=cfgs.three_d,
                    test_data=eval_dl, 
                    face_db=noise_db,
                    fr=eval_fr,
                    device=device,
                    bin_mask=cfgs.bin_mask,
                    epsilon=cfgs.epsilon,
                    masks=univ_mask.repeat(img_num, 1, 1, 1), 
                    query_portion=cfgs.query_portion,
                    vis_eval=cfgs.vis_eval,
                    lpips_backbone=cfgs.lpips_backbone, 
                    verbose=True, 
                    replication_attack=cfgs.replication_attack,
                    uvmapper=uvmapper,
                    replication_mask=rep_mask)
                if mode == 'train':
                    res['training_time'] = training_time
                with open(os.path.join(res_save_path, f"eval_res_{eval_fr.model_name}.yaml"), 'w') as f:
                    yaml.dump(res, f)
                if cfgs.eval_compression:
                    noise_db = build_compressed_face_db(db_path=noise_db_path, 
                                                        fr=eval_fr, 
                                                        device=device, 
                                                        compression_methods=cfgs.eval_compression_methods, 
                                                        compression_cfgs=cfgs.compression_cfgs)
                    compression_res = compression_eval(
                        compression_methods=cfgs.eval_compression_methods, 
                        compression_cfgs=cfgs.compression_cfgs,
                        three_d=cfgs.three_d,
                        test_data=eval_dl,
                        face_db=noise_db,
                        fr=eval_fr,
                        device=device,
                        bin_mask=cfgs.bin_mask,
                        epsilon=cfgs.epsilon,
                        masks=univ_mask.repeat(img_num, 1, 1, 1),
                        query_portion=cfgs.query_portion
                    )
                    with open(os.path.join(res_save_path, f"compression_res_{eval_fr.model_name}.yaml"), 'w') as f:
                        yaml.dump(compression_res, f)
            del eval_dl
            complete_del()
            if cfgs.end2end_eval:
                end2end_res = eval_mask_end2end(
                                three_d=cfgs.three_d,
                                test_raw_imgs=load_imgs(img_paths=[f for f_idx, f in enumerate(protectee_data['eval']) if f_idx not in no_face], 
                                                        img_sz=-1, usage_portion=1.0, drange=1, device=device), 
                                face_db_path=noise_db_path, 
                                frs=eval_frs,
                                fd=fd,
                                uvmapper=uvmapper, 
                                device=device,
                                bin_mask=cfgs.bin_mask,
                                epsilon=cfgs.epsilon,
                                mask=univ_mask.clone(), 
                                query_portion=cfgs.query_portion,
                                strict_crop=cfgs.strict_crop,
                                resize_face=cfgs.resize_face, 
                                jpeg=cfgs.jpeg, 
                                smoothing=None if cfgs.smoothing.lower() == 'none' else cfgs.smoothing.lower(),
                                vis_eval=False, 
                                lpips_backbone=cfgs.lpips_backbone,
                                verbose=True, 
                                eval_scene1_db=None, 
                                replication_attack=cfgs.replication_attack,
                                replication_mask=rep_mask)
                for fr_name, prot_res in end2end_res['prot_results'].items():
                    with open(os.path.join(res_save_path, f"end2end_eval_res_{fr_name}.yaml"), 'w') as f:
                        yaml.dump(prot_res, f)
    if mode == 'train':
        print(f"Average training time: {np.mean(training_times):.2f} seconds.")

    if cfgs.eval_scene == 1:
        with torch.no_grad():
            protectee_eval_features = {}
            pbar = tqdm.tqdm(protectee_eval_data.items(), desc="Preparing evaluation features for all protectees")
            for protectee, protectee_data in pbar:
                pbar.set_postfix({"Protectee": protectee})
                eval_imgs = protectee_data['imgs'].to(device)
                eval_uvs = protectee_data['uvs'].to(device)
                eval_bin_masks = protectee_data['bin_masks'].to(device)
                protectee_mask = protectee_masks[protectee].to(device)
                if cfgs.three_d:
                    perts = torch.clamp(F.grid_sample(protectee_mask.repeat(eval_imgs.shape[0], 1, 1, 1), eval_uvs, align_corners=True, mode='bilinear'), -cfgs.epsilon, cfgs.epsilon)
                else:
                    perts = torch.clamp(protectee_mask.repeat(eval_imgs.shape[0], 1, 1, 1), -cfgs.epsilon, cfgs.epsilon)
                if cfgs.bin_mask:
                    perts *= eval_bin_masks
                prot_imgs = torch.clamp(eval_imgs + perts, 0., 1.)
                orig_feature_num = int(eval_imgs.shape[0] * cfgs.query_portion)
                protectee_feature = {}
                for eval_fr in eval_frs:
                    orig_features = eval_fr(eval_imgs)[:orig_feature_num]
                    prot_features = eval_fr(prot_imgs)[orig_feature_num:]
                    protectee_feature[eval_fr.model_name] = {}
                    protectee_feature[eval_fr.model_name]['uncompressed'] = torch.cat([orig_features, prot_features], dim=0).cpu()
                    if cfgs.eval_compression:
                        protectee_feature[eval_fr.model_name]['compressed'] = {}
                        for compression_method in cfgs.eval_compression_methods:
                            compressed_orig_features = eval_fr(compress(eval_imgs, method=compression_method, differentiable = False, **cfgs.compression_cfgs[compression_method])).cpu()[:orig_feature_num]
                            compressed_prot_features = eval_fr(compress(prot_imgs, method=compression_method, differentiable = False, **cfgs.compression_cfgs[compression_method])).cpu()[orig_feature_num:]
                            protectee_feature[eval_fr.model_name]['compressed'][compression_method] = torch.cat([compressed_orig_features, compressed_prot_features], dim=0)
                protectee_eval_features[protectee] = protectee_feature
            for protectee_idx, (protectee, protectee_data) in enumerate(protectee_eval_data.items()):
                print("\n"+"#" * 50)
                print(f"Evaluating under Eval Scene {cfgs.eval_scene} for Protectee: {protectee} ({protectee_idx+1}/{len(data)})")
                print("#" * 50+ "\n")
                res_save_path = os.path.join(res_path, protectee)
                eval_imgs = protectee_data['imgs'].to(device)
                eval_uvs = protectee_data['uvs'].to(device)
                eval_bin_masks = protectee_data['bin_masks'].to(device)
                img_num = eval_imgs.shape[0]
                protectee_mask = protectee_masks[protectee].to(device)
                eval_dl = DataLoader(
                    dataset=TensorDataset(eval_imgs, eval_uvs, eval_bin_masks),
                    batch_size=16,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False
                )
                if cfgs.replication_attack:
                    mask_path = os.path.join(BASE_PATH, 'experiments', cfgs.replication_mask_name[0], protectee, cfgs.replication_mask_name[1])
                    rep_mask = load_mask(mask_path=mask_path, device=device)
                else:
                    rep_mask = None
                for eval_fr in eval_frs:
                    noise_db = build_facedb(db_path=noise_db_path, fr=eval_fr, device=device)
                    for other_protectee, other_protectee_features in protectee_eval_features.items():
                        if other_protectee == protectee:
                            continue
                        noise_db[other_protectee] = other_protectee_features[eval_fr.model_name]['uncompressed'].to(device)
                    res = eval_masks(
                        three_d=cfgs.three_d,
                        test_data=eval_dl, 
                        face_db=noise_db,
                        fr=eval_fr,
                        device=device,
                        bin_mask=cfgs.bin_mask,
                        epsilon=cfgs.epsilon,
                        masks=protectee_mask.repeat(img_num, 1, 1, 1), 
                        query_portion=cfgs.query_portion,
                        vis_eval=cfgs.vis_eval,
                        lpips_backbone=cfgs.lpips_backbone, 
                        verbose=True, 
                        replication_attack=cfgs.replication_attack,
                        uvmapper=uvmapper,
                        replication_mask=rep_mask)
                    if mode == 'train':
                        res['training_time'] = training_times[protectee_idx]
                    with open(os.path.join(res_save_path, f"eval_res_{eval_fr.model_name}.yaml"), 'w') as f:
                        yaml.dump(res, f)
                    if cfgs.eval_compression:
                        noise_db = build_compressed_face_db(db_path=noise_db_path, 
                                                            fr=eval_fr, 
                                                            device=device, 
                                                            compression_methods=cfgs.eval_compression_methods, 
                                                            compression_cfgs=cfgs.compression_cfgs)
                        for other_protectee, other_protectee_features in protectee_eval_features.items():
                            if other_protectee == protectee:
                                continue
                            for compression_method in cfgs.eval_compression_methods:
                                noise_db[compression_method][other_protectee] = other_protectee_features[eval_fr.model_name]['compressed'][compression_method].to(device)
                        compression_res = compression_eval(
                            compression_methods=cfgs.eval_compression_methods, 
                            compression_cfgs=cfgs.compression_cfgs,
                            three_d=cfgs.three_d,
                            test_data=eval_dl,
                            face_db=noise_db,
                            fr=eval_fr,
                            device=device,
                            bin_mask=cfgs.bin_mask,
                            epsilon=cfgs.epsilon,
                            masks=protectee_mask.repeat(img_num, 1, 1, 1),
                            query_portion=cfgs.query_portion
                        )
                        with open(os.path.join(res_save_path, f"compression_res_{eval_fr.model_name}.yaml"), 'w') as f:
                            yaml.dump(compression_res, f)
                del eval_dl
                complete_del()
                if cfgs.end2end_eval:
                    eval_scene1_db = {}
                    for eval_fr in eval_frs:
                        eval_scene1_db[eval_fr.model_name] = {}
                        for other_protectee, other_protectee_features in protectee_eval_features.items():
                            if other_protectee == protectee:
                                continue
                            eval_scene1_db[eval_fr.model_name][other_protectee] = other_protectee_features[eval_fr.model_name]['uncompressed'].to(device)
                    end2end_res = eval_mask_end2end(
                                three_d=cfgs.three_d,
                                test_raw_imgs=load_imgs(img_paths=[f for f_idx, f in enumerate(data[protectee]['eval']) if f_idx not in protectee_no_face_imgs[protectee]], 
                                                        img_sz=-1, usage_portion=1.0, drange=1, device=device), 
                                face_db_path=noise_db_path, 
                                frs=eval_frs,
                                fd=fd,
                                uvmapper=uvmapper, 
                                device=device,
                                bin_mask=cfgs.bin_mask,
                                epsilon=cfgs.epsilon,
                                mask=protectee_mask.clone(), 
                                query_portion=cfgs.query_portion,
                                strict_crop=cfgs.strict_crop,
                                resize_face=cfgs.resize_face, 
                                jpeg=cfgs.jpeg, 
                                smoothing=None if cfgs.smoothing.lower() == 'none' else cfgs.smoothing.lower(),
                                vis_eval=False, 
                                lpips_backbone=cfgs.lpips_backbone,
                                verbose=True, 
                                eval_scene1_db=eval_scene1_db, 
                                replication_attack=cfgs.replication_attack,
                                replication_mask=rep_mask)
                    for fr_name, prot_res in end2end_res['prot_results'].items():
                        with open(os.path.join(res_save_path, f"end2end_eval_res_{fr_name}.yaml"), 'w') as f:
                            yaml.dump(prot_res, f)