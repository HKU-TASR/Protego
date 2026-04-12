import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
import argparse
import sys
import math
import random
import itertools
import copy

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from omegaconf import OmegaConf

from protego.utils import get_usable_img_paths
from protego.protego_train_robust import train_protego_mask_robust
from protego.protego_train import train_protego_mask
from protego.advcloak import AdvCloakGenerator
from protego.run_exp import run
from protego.FacialRecognition import BASIC_POOL, SPECIAL_POOL, VIT_FAMILY
from protego import BASE_PATH

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    args = argparse.ArgumentParser()
    args.add_argument('--exp_name', type=str, default='simple_cosine', help='The name of the experiment.')
    args.add_argument('--device', type=str, help='The device to use for training. (cpu, mps, cuda:0, etc.)')
    args = args.parse_args()

    configs = {
        # Running env
        'global_random_seed' : 42, 
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
        'batch_size' : 4,  
        'epsilon' : 16 / 255., 
        'min_ssim' : 0.95, 
        'learning_rate' : 0.01 * (16 / 255.), # 0.01 * (16 / 255.)
        'mask_size' : 224, 
        'mask_random_seed': 114, 
        'bin_mask': True, # Whether to use binary mask. If True, the perturbation will be restricted to the face area. 
        'train_fr_names': [n for n in BASIC_POOL if n != 'ir50_adaface_casia'],

        # Eval configs
        'eval_scene': 1,
        'mask_name': ['vggface2_emotions', 'univ_mask.npy'], 
        'eval_db': 'face_scrub',
        'eval_fr_names': ['ir50_adaface_casia'],
        'save_univ_mask': True, 
        'visualize_interval': 30,
        'query_portion': 0.5,
        'vis_eval': True, 
        'lpips_backbone': "vgg", 
        'replication_attack': False, 
        'replication_mask_name': ['default', 'univ_mask.npy'],
        'input_robustness_eval': False,
        'input_robustness_methods': ['occlude'],
        'input_robustness_cfgs' : {
            'gaussian': {
                'kernel_size': 9, 
                'sigma': 2.0,
            }, 
            'median': {
                'kernel_size': 9
            }, 
            'jpeg': {
                'quality': 70, 
            }, 
            'resize': {
                'resz_percentage': 0.2,
                'mode': 'bilinear'
            }, 
            'quantize': {
                'precision': 'uint8'
            }, 
            'vid_codec': {
                'codec': 'h264',
                'crf': 32,
                'preset': 'faster'
            },
            'occlude': {
                'occ_size': [64, 64],
                'size_mode': 'same_dim',
                'occ_val': 0.,
                'occ_mode': 'center_bottom',
                'sharpness': 20.0
            }
        },
        'end2end_eval': False, 
        'strict_crop': True, 
        'resize_face': False, 
        'jpeg': True, 
        'smoothing': 'none', # Options: 'gaussian', 'median', 'none'
        'eval_compression': False, # Whether to evaluate the compression of the mask.
        'eval_compression_methods': ['none', 'gaussian', 'median', 'jpeg', 'resize', 'quantize', 'vid_codec'], # The compression methods to evaluate.
        'compression_cfgs' : {
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
        },
        'train_compression_method': ['gaussian', 'median', 'jpeg', 'quantize', 'resize'],
        'train_compression_cfgs' : {
            # Gaussian Filter
            'gaussian': {
                'kernel_size': 9, 
                'sigma': 2.0
            }, 
            # Median Filter
            'median': {
                'kernel_size': 9
            }, 
            'resize': {
                'resz_percentage': 0.4,
                'mode': 'bicubic'
            },
            # JPEG Compression
            'jpeg': {
                'quality': 70
            }, 
            # Quantize
            'quantize': {
                'precision': 'uint8',
                'diff_method': 'ste'
            }
        }
    }
    ####################################################################################################################
    # Run
    ####################################################################################################################
    for method, kwargs in configs['train_compression_cfgs'].items():
        kwargs['differentiable'] = True
    cfgs = OmegaConf.create(configs)
    cfgs.exp_name = args.exp_name if '--exp_name' in sys.argv else cfgs.exp_name
    cfgs.device = args.device if '--device' in sys.argv else cfgs.device
    torch.manual_seed(cfgs.global_random_seed)
    random.seed(cfgs.global_random_seed)

    train_portion = cfgs.train_portion
    shuffle_data = cfgs.shuffle
    usage_portion = 1.

    """rep_train_num = 100
    front_data_path = os.path.join(BASE_PATH, 'face_db', 'BC_front')
    left_data_path = os.path.join(BASE_PATH, 'face_db', 'BC_left')
    right_data_path = os.path.join(BASE_PATH, 'face_db', 'BC_right')
    eval_data_path = os.path.join(BASE_PATH, 'face_db', 'face_scrub')
    protectees = ["Bradley_Cooper"]
    data = {}
    for protectee in protectees:
        front_imgs = [os.path.join(front_data_path, img_name) for img_name in os.listdir(front_data_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
        left_imgs = [os.path.join(left_data_path, img_name) for img_name in os.listdir(left_data_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
        right_imgs = [os.path.join(right_data_path, img_name) for img_name in os.listdir(right_data_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
        #eval_imgs = get_usable_img_paths(os.path.join(eval_data_path, protectee))
        #eval_imgs = eval_imgs[math.floor(len(eval_imgs) * train_portion):]
        #print(len(eval_imgs))

        rep_train_imgs = right_imgs+left_imgs+front_imgs
        #random.shuffle(rep_train_imgs)
        #rep_train_imgs = rep_train_imgs[:rep_train_num]
        #if len(rep_train_imgs) < rep_train_num: 
        #    print(f"Using {rep_train_num - len(rep_train_imgs)} duplicated images for replication attack training for {protectee}.")
        #    rep_train_imgs += rep_train_imgs[:(rep_train_num - len(rep_train_imgs))]
        data[protectee] = {'train': rep_train_imgs, 'eval': rep_train_imgs}"""
    """train_data_path = os.path.join(BASE_PATH, 'face_db', 'vggface2_emotions_cropped')
    emotion_data_path = os.path.join(BASE_PATH, 'face_db', 'vggface2_emo_fear_cropped')
    protectees = sorted([name for name in os.listdir(emotion_data_path) if not name.startswith(('.', '_'))])
    data = {}
    for protectee in protectees:
        protectee_path = os.path.join(train_data_path, protectee)
        imgs = get_usable_img_paths(protectee_path)
        train_num = math.floor(len(imgs) * train_portion)
        #train_num = min(train_num, 30)
        #eval_num = len(imgs) - train_num
        eval_num = min(23, len(imgs) - train_num)
        #data[protectee] = {'train': imgs[:train_num], 'eval': imgs[train_num:train_num+eval_num]}
        eval_imgs = imgs[train_num:train_num+eval_num]
        protectee_path = os.path.join(emotion_data_path, protectee)
        imgs = get_usable_img_paths(protectee_path)
        eval_num = min(23, len(imgs))
        eval_imgs = imgs[:eval_num] + eval_imgs
        print(f"{protectee}: {len(eval_imgs)}")
        data[protectee] = {'eval': eval_imgs}"""
    train_data_path = os.path.join(BASE_PATH, 'face_db', 'bfw_gender_woman_cropped')
    protectees = sorted([name for name in os.listdir(train_data_path) if not name.startswith(('.', '_'))])
    data = {}
    for protectee in protectees:
        protectee_path = os.path.join(train_data_path, protectee)
        imgs = get_usable_img_paths(protectee_path)
        train_num = math.floor(len(imgs) * train_portion)
        data[protectee] = {'train': imgs[:train_num], 'eval': imgs[train_num:]}
    run(cfgs, mode='train', data=data, train=train_protego_mask)
    #run(cfgs, mode='eval', data=data)
    """eval_data_path = os.path.join(BASE_PATH, 'face_db', 'fs_uncropped')
    protectees = sorted([name for name in os.listdir(eval_data_path) if not name.startswith(('.', '_'))])
    data = {}
    for protectee in protectees:
        protectee_path = os.path.join(eval_data_path, protectee)
        imgs = [os.path.join(protectee_path, img_name) for img_name in os.listdir(protectee_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
        if shuffle_data:
            rand_gen = torch.Generator()
            rand_gen.manual_seed(cfgs.global_random_seed)
            indices = torch.randperm(len(imgs), generator=rand_gen).tolist()
            imgs = [imgs[i] for i in indices]
        usage_num = math.floor(len(imgs) * usage_portion)
        data[protectee] = {'eval': imgs[:usage_num]}
    #run(cfgs, mode='train', data=data, train=train_protego_mask_robust)
    run(cfgs, mode='eval', data=data)"""