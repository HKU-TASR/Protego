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
        'global_random_seed': 42, # Random seed for all random operations (data shuffling, model initialization, etc.) to ensure reproducibility.
        'device': 'cuda:0', # The device to use. Will be overridden by the command line argument --device if provided.
        'exp_name': 'default', # The name of the experiment. Will be used to create a subfolder under experiments/ to save the results. Will be overridden by the command line argument --exp_name if provided.

        # Training data
        'train_portion': 0.6, # The portion of images to use for training. The rest will be used for evaluation.
        'uv_gen_align_ldmk': False, # ! Don't change
        'uv_gen_batch': 8, # UV map generation batch size. Set to a larger number to speed up UV map generation, but it will consume more memory.
        'need_cropping': False, # Set to True if the images are not tightly cropped around the face. Set to False otherwise.
        'fd_name': 'mtcnn', # The face detector to use for cropping. Only used when need_cropping is True. (mtcnn, retinaface, mediapipe, etc.)
        'crop_loosen': 1., # The looseness of cropping. Only used when need_cropping is True. 1 is recommended.
        'shuffle': False,

        # Training configs
        'three_d': True, # ! Don't change.
        'epoch_num': 100,
        'batch_size': 16,
        'epsilon': 16 / 255., # The maximum perturbation magnitude. 16/255 is recommended for a good balance between effectiveness and imperceptibility.
        'min_ssim': 0.95, # The minimum required SSIM between the original image and the perturbed image. Set to a higher value for better visual quality, but it may reduce the effectiveness of the perturbation.
        'learning_rate': 0.01 * (16 / 255.), # The learning rate for training. We recommend setting it to 0.01 times the epsilon value for stable training.
        'mask_size': 224, # ! Don't change.
        'mask_random_seed': 114,
        'bin_mask': True,  # Whether to restrict the perturbation to the face area.
        'train_fr_names': [n for n in BASIC_POOL if n != 'ir50_adaface_casia'], # Surrogate FR models to use for training.

        # Eval configs
        'mask_name': ['default', 'univ_mask.npy'],
        'eval_db': 'face_scrub',
        'eval_fr_names': ['ir50_adaface_casia'],
        'save_univ_mask': True, # ! Don't change.
        'visualize_interval': 15, # The interval (in epochs) at which to visualize the results.
        'query_portion': 0.5, # ! Don't change.
        'vis_eval': True, # Whether to evaluate the visual quality of the perturbed images.
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

    # To evaluate previously trained PPTs (located under experiments/<mask_name[0]>/<protectee>/<mask_name[1]>) without retraining, use the eval mode instead:
    # run(cfgs, mode='eval', data=data)
