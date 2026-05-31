import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
import argparse
import sys
import math

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from omegaconf import OmegaConf

from protego.utils import get_usable_img_paths
from protego.protego_train import train_protego_mask
from protego.run_exp import run
from protego.FacialRecognition import BASIC_POOL
from protego import BASE_PATH

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    args = argparse.ArgumentParser()
    args.add_argument('--exp_name', type=str, default='default', help='The name of the experiment.')
    args.add_argument('--device', type=str, help='The device to use. (cpu, mps, cuda:0, etc.)')
    args = args.parse_args()

    configs = {
        # Running env
        'global_random_seed': 42,
        'device': 'cuda:0',
        'exp_name': 'default',

        # Training data
        'train_portion': 0.6,
        'uv_gen_align_ldmk': False,
        'uv_gen_batch': 8,
        'need_cropping': False,
        'fd_name': 'mtcnn',
        'crop_loosen': 1.,
        'shuffle': False,

        # Training configs
        'three_d': True,
        'epoch_num': 100,
        'batch_size': 16,
        'epsilon': 16 / 255.,
        'min_ssim': 0.95,
        'learning_rate': 0.01 * (16 / 255.),
        'mask_size': 224,
        'mask_random_seed': 114,
        'bin_mask': True,  # Whether to restrict the perturbation to the face area.
        'train_fr_names': [n for n in BASIC_POOL if n != 'ir50_adaface_casia'],

        # Eval configs
        'mask_name': ['default', 'univ_mask.npy'],
        'eval_db': 'face_scrub',
        'eval_fr_names': ['ir50_adaface_casia'],
        'save_univ_mask': True,
        'visualize_interval': 10,
        'query_portion': 0.5,
        'vis_eval': True,
        'lpips_backbone': "vgg",
    }
    ####################################################################################################################
    # Run
    ####################################################################################################################
    cfgs = OmegaConf.create(configs)
    cfgs.exp_name = args.exp_name if '--exp_name' in sys.argv else cfgs.exp_name
    cfgs.device = args.device if '--device' in sys.argv else cfgs.device
    torch.manual_seed(cfgs.global_random_seed)

    train_portion = cfgs.train_portion
    shuffle_data = cfgs.shuffle
    train_data_path = os.path.join(BASE_PATH, 'face_db', 'face_scrub')
    protectees = sorted([name for name in os.listdir(train_data_path) if not name.startswith(('.', '_'))])
    data = {}
    for protectee in protectees:
        protectee_path = os.path.join(train_data_path, protectee)
        imgs = get_usable_img_paths(protectee_path)
        train_num = math.floor(len(imgs) * train_portion)
        if shuffle_data:
            rand_gen = torch.Generator()
            rand_gen.manual_seed(cfgs.global_random_seed)
            indices = torch.randperm(len(imgs), generator=rand_gen).tolist()
            imgs = [imgs[i] for i in indices]
        data[protectee] = {'train': imgs[:train_num], 'eval': imgs[train_num:]}

    # Train a PPT for every protectee, then run the cross-protectee retrieval evaluation.
    run(cfgs, mode='train', data=data, train=train_protego_mask)

    # To evaluate previously trained PPTs (located under experiments/<mask_name[0]>/<protectee>/<mask_name[1]>)
    # without retraining, use the eval mode instead:
    # run(cfgs, mode='eval', data=data)
