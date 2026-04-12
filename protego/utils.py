import os
from typing import Dict, List, Tuple, Any, Optional, Union
import gc
import sys
import filelock

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import PIL
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_msssim import ssim as calc_ssim
import lpips

from .FaceDetection import FD
from .FacialRecognition import FR
from .compression import compress
from . import BASE_PATH
from .UVMapping import UVGenerator

def kmeans(features: torch.Tensor, n_clusters: int, rand_seed: int = 42, max_iter: int = 1000, distance: str = "cosine") -> torch.Tensor:
    """
    Use torch to accelerate k-means clustering.

    Args:
        features (torch.Tensor): The features to cluster. Shape: (N, D), where N is the number of features and D is the feature dimension.
        n_clusters (int): The number of clusters.
        rand_seed (int): The random seed for initialization.
        distance (str): The distance metric to use. Either 'cosine' or 'euclidean'.

    Returns:
        torch.Tensor: The cluster assignments for each feature. Shape: (N,), where each value is in [0, n_clusters-1].
    """
    # Initialization
    rand_generator = torch.Generator().manual_seed(rand_seed)
    centroids = features[torch.randperm(features.size(0), generator=rand_generator)[:n_clusters]]
    for _ in range(max_iter):
        if distance == 'cosine':
            sim_matrix = torch.matmul(F.normalize(features, p=2, dim=1), F.normalize(centroids, p=2, dim=1).T)
            preds = sim_matrix.argmax(dim=1)
        elif distance == 'euclidean':
            dist_matrix = torch.cdist(features, centroids, p=2)
            preds = dist_matrix.argmin(dim=1)
        else:
            raise ValueError(f"Unsupported distance function: {distance}. Use 'cosine' or 'euclidean'.")
        new_centroids = torch.stack([features[preds == i].mean(dim=0) if torch.any(preds == i) else centroids[i] for i in range(n_clusters)], dim=0)
        if torch.all(centroids == new_centroids):
            return preds
        centroids = new_centroids
    print(f'Warning: k-means did not converge within {max_iter} iterations.')
    return preds

def crop_face(img: torch.Tensor, 
            detector: FD, 
            conf_thresh: float = 0.8, 
            min_h: int = 20, 
            min_w: int = 20, 
            crop_loosen: float = 1.0,
            verbose: bool = True, 
            strict: bool = False) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int, int, int]]]:
    """
    Crop the face from the image using FD.

    Args:
        img (torch.Tensor): FloatTensor. Range [0, 255] or [0, 1]. Shape [3, H, W]. RGB.
        conf_thresh (float): Confidence threshold for face detection. Not passed to the detection model, used only in post-processing.
        min_h (int): Minimum height of the detected face.
        min_w (int): Minimum width of the detected face.
        crop_loosen (float): Factor to loosen the crop around the detected face. 1.0 means no change.
        verbose (bool): Whether to print warnings.
        strict (bool): Whether to return None, None if warnings are raised.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int, int, int]]: The cropped face (Range is the same as input, RGB, [3, H, W]) and its position in the original image.
    """
    det_res = detector(img.float())
    if det_res is None:
        return None, None
    bboxs = [list(det_res[i][:5]) for i in range(len(det_res))]
    height, width = img.shape[1:]
    cropped_face = None
    position = None
    largest_area = 0
    for bbox in bboxs:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        conf = bbox[4] 
        if conf < conf_thresh:  # Confidence threshold
            if verbose:
                print(f"Discarding face with low confidence: {conf:.2f}")
            continue
        face_width = x2 - x1
        face_height = y2 - y1
        if face_height < min_h or face_width < min_w:
            if verbose:
                print(f"Discarding face with too small dimensions: width={face_width}, height={face_height}")
            continue
        face_area = face_width * face_height
        if face_area < largest_area:
            continue
        largest_area = face_area

        square_size = max(face_width, face_height)
        if crop_loosen != 1.0:
            square_size = int(square_size * crop_loosen)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        full_size = min(square_size, width, height)
        if full_size == width:
            if verbose:
                print(f"Warning: Original Image too narrow, cannot crop face properly.")
            if strict:
                return None, None
            square_x1, square_x2 = 0, width
            half_size = width // 2
            square_y1 = center_y - half_size
            square_y2 = center_y + half_size
        elif full_size == height:
            if verbose:
                print(f"Warning: Original Image too short, cannot crop face properly.")
            if strict:
                return None, None
            square_y1, square_y2 = 0, height
            half_size = height // 2
            square_x1 = center_x - half_size
            square_x2 = center_x + half_size
        elif full_size == square_size:
            half_size = full_size // 2
            square_x1 = center_x - half_size
            square_y1 = center_y - half_size
            square_x2 = center_x + half_size
            square_y2 = center_y + half_size

        cropped_face = img[:, square_y1:square_y2, square_x1:square_x2].clone()
        position = (square_x1, square_y1, square_x2, square_y2)
    if cropped_face is None or position is None:
        if verbose:
            print(f"Warning: No valid face detected that meets the criteria. conf_thresh: {conf_thresh}, min_h: {min_h}, min_w: {min_w}.")
        return None, None
    if cropped_face.shape[1] < min_h or cropped_face.shape[2] < min_w:
        if verbose:
            print(f"Warning: No valid face detected that meets the criteria. min_h: {min_h}, min_w: {min_w}.")
        return None, None
    x1, y1, x2, y2 = position
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        if verbose:
            print(f"Warning: Cropped face goes out of image boundary. Image size: ({width}, {height}). Cropped position: ({x1}, {y1}, {x2}, {y2}).")
        return None, None
    return cropped_face, position

def complete_del() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps"):
        is_available = getattr(torch.mps, "is_available", None)
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(is_available) and callable(empty_cache):
            try:
                if is_available():
                    empty_cache()
            except AttributeError:
                # Older PyTorch builds may expose torch.backends.mps without torch.mps helpers
                pass

def img2tensor(img: np.ndarray, drange: int = 1, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Convert a numpy image to a tensor.

    Args:
        img (np.ndarray): The numpy image to convert. Shape: (H, W, 3), dtype: uint8, range [0, 255].
        drange (int): The range of the image. Default is 1. Either 1 or 255.
        device (torch.device): The device to store the tensor. Default is CPU.

    Returns:
        torch.Tensor: The converted tensor. Shape: (1, 3, H, W), dtype: float32., range [0, 1] or [0, 255], depending on drange.
    """
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device).to(device)
    img_tensor = img_tensor.permute(2, 0, 1)
    if drange == 1:
        img_tensor /= 255. 
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def load_mask(mask_path: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Load a universal mask from a .npy file.

    Args:
        mask_path (str): The path to the .npy file containing the mask.
        device (torch.device): The device to store the tensor. Default is CPU.

    Returns:
        torch.Tensor: The loaded mask tensor. Shape: (1, 3, H, W), dtype: float32., range [0, 1].
    """
    return torch.tensor(np.load(mask_path, allow_pickle=True)[0], dtype=torch.float32, device=device).to(device).unsqueeze(0)

def load_imgs(base_dir: str = None, img_paths: Optional[List[str]] = None, img_sz: int = 224, usage_portion: float = 1.0, drange: int = 1, device: torch.device = torch.device('cpu'), return_img_paths: bool = False) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[str]], Tuple[List[torch.Tensor], List[str]]]:
    """
    Read images from a directory and convert them to tensors. If base dir is given, only imgs in ['jpg', 'png', 'jpeg', 'bmp'] format and NOT starting with '.' or '_' will be read.

    Args:
        base_dir (str): The directory containing the images.
        img_paths (List[str]): The list of image paths. If provided, base_dir will be ignored.
        img_sz (int): The size of the image. If set to -1, it will not be resized. 
        usage_portion (float): The portion of the images to use.
        drange (int): The range of the image. Default is 1. Either 1 or 255.
        device (torch.device): The device to store the tensor. Default is CPU.
        return_img_paths (bool): Whether to return the list of image paths.

    Returns:
        Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[str]], Tuple[List[torch.Tensor], List[str]]]: The tensor of the images. Shape [B, 3, img_sz, img_sz]. RGB, range [0, 1], dtype: float32.
        A list of tensors with shape [1, 3, H, W] if img_sz is -1. Will also return the list of image paths if return_img_paths is True.
    """
    img_tensors = []
    valid_img_paths = []
    if base_dir is not None and img_paths is None:
        img_num = len(os.listdir(base_dir))
        imgs_names = sorted([os.path.join(base_dir, name) for name in os.listdir(base_dir) if name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not (name.startswith('.') or name.startswith('_'))])
    elif img_paths is not None and base_dir is None:
        img_num = len(img_paths)
        imgs_names = img_paths
    else:
        raise ValueError("Either base_dir or img_paths should be provided, but not both or neither.")
    for img_idx, img_path in enumerate(imgs_names):
        try:
            img = PIL.Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to open image {img_path}. Skipping. Error: {e}")
            continue
        img = np.array(img)
        if img_sz != -1:
            img = cv2.resize(img, (img_sz, img_sz), interpolation=cv2.INTER_LANCZOS4 if img.shape[0] < img_sz else cv2.INTER_AREA)
        img_tensors.append(img2tensor(img, drange=drange, device=device))
        if return_img_paths:
            valid_img_paths.append(img_path)
        if len(img_tensors) >= int(img_num * usage_portion):
            break
    imgs = torch.cat(img_tensors, dim=0) if img_sz != -1 else img_tensors
    if return_img_paths:
        return imgs, valid_img_paths
    return imgs

def preextract_features(base_path: str, fr: FR, device: torch.device, save_name: str, compression_cfg: Optional[Tuple[str, Dict[str, Any]]] = None) -> None:
    """
    A unified gateway to pre-extract features for all images in the given directory.

    Args:
        base_path (str): The path to the database directory.
        fr (FR): The facial recognition model.
        device (torch.device): The device to use for computation.
        save_name (str): The name to save the extracted features.
        compression_cfg (Optional[Dict[str, Any]]): The compression configuration. If provided, the images will be compressed before feature extraction.
    """
    usable_imgs = os.path.join(base_path, 'imgs_list.txt')
    if os.path.exists(usable_imgs):
        img_names = get_usable_img_paths(base_path)
        imgs = load_imgs(img_paths=img_names, img_sz=224, usage_portion=1., drange=1, device=device)
    else:
        imgs, img_names = load_imgs(base_dir=base_path, img_sz=224, usage_portion=1., drange=1, device=device, return_img_paths=True)
    if compression_cfg is not None:
        imgs = compress(imgs=imgs, method=compression_cfg[0], differentiable=False, **compression_cfg[1])
    lockfile = filelock.FileLock(os.path.join(base_path, f"{save_name}.lock"))
    with lockfile:
        torch.save(fr(imgs).detach().cpu(), os.path.join(base_path, save_name))
    lockfile.release()
    if not os.path.exists(usable_imgs):
        with open(usable_imgs, 'w') as f:
            for img_name in img_names:
                f.write(os.path.relpath(img_name, base_path) + '\n')

def get_usable_img_paths(base_dir: str) -> List[str]:
    imgs = []
    with open(os.path.join(base_dir, 'imgs_list.txt'), 'r') as f:
        for line in f.readlines():
            imgs.append(os.path.join(base_dir, line.strip()))
    return imgs

def build_facedb(db_path: str, fr: Union[str, FR], device: torch.device, return_img_paths: bool = False) -> Union[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, List[str]]]]:
    """
    Build a database of features from the images in the given directory. Folders starting with '.' or '_' will be ignored.

    Args:
        db_path (str): The path to the database directory.
        fr (Union[str, FR]): The facial recognition model or the name of the model to use.
        device (torch.device): The device to use for computation.
        return_img_paths (bool): Whether to return the list of image paths along with the features.

    Returns:
        Union[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, List[str]]]]: A dictionary containing the features for each person in the database.
         - Dict[str, torch.Tensor]: A dictionary containing the features for each person in the database.
         - Dict[str, Tuple[torch.Tensor, List[str]]]: A dictionary containing the features and image paths for each person in the database if return_img_paths is True.
    """
    db = {}
    names = sorted([n for n in os.listdir(db_path) if not n.startswith(('.', '_'))])
    fr_name = fr if isinstance(fr, str) else fr.model_name
    fr_model = None
    for name in names:
        personal_path = os.path.join(db_path, name)
        feature_path = os.path.join(personal_path, f'{fr_name}.pt')
        try:
            features = torch.load(feature_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"{fr_name} features of {name} not found or failed to load. (Re)Extracting features. Error: {e}")
            if isinstance(fr, str) and fr_model is None:
                fr_model = FR(model_name=fr, device=device)
            elif isinstance(fr, FR) and fr_model is None:
                fr_model = fr
            preextract_features(base_path=personal_path, fr=fr_model, device=device, save_name=f'{fr_name}.pt')
            features = torch.load(feature_path, map_location=device, weights_only=False)
        db[name] = features.to(device)
        if return_img_paths:
            img_paths = get_usable_img_paths(personal_path)
            db[name] = (features.to(device), img_paths)
    return db

def build_compressed_face_db(db_path: str, fr: Union[str, FR], device: torch.device, compression_methods: List[str], compression_cfgs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Build a database of features from the compressed images in the given directory.

    Args:
        db_path (str): The path to the database directory.
        fr (Union[str, FR]): The facial recognition model or the name of the model to use.
        device (torch.device): The device to use for computation.
        compression_methods (List[str]): The list of compression methods to use.
        compression_cfgs (Dict[str, Dict[str, Any]]): The configurations for each compression method.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: {'compression_method': {'name': features}}
    """
    with torch.no_grad():
        db = {compression_method: {} for compression_method in compression_methods}
        pbar = tqdm.tqdm(os.listdir(db_path), desc="Building compressed face database")
        fr_name = fr if isinstance(fr, str) else fr.model_name
        fr_model = None
        for name in pbar:
            if name.startswith('.') or name.startswith('_'):
                continue
            personal_path = os.path.join(db_path, name)
            for method in compression_methods:
                cfgs_str = ''
                for k, v in compression_cfgs[method].items():
                    cfgs_str += f"_{k}_{v}"
                f_name = os.path.join(personal_path, f"{fr_name}_{method}{cfgs_str}.pt")
                try:
                    db[method][name] = torch.load(f_name, map_location=device, weights_only=False)
                except Exception as e:
                    print(f"{fr_name} features for {name} with compression {method} not found or failed to load. (Re)Extracting features. Error: {e}")
                    if isinstance(fr, str) and fr_model is None:
                        fr_model = FR(model_name=fr, device=device)
                    elif isinstance(fr, FR) and fr_model is None:
                        fr_model = fr
                    preextract_features(base_path=personal_path, fr=fr_model, device=device, save_name=os.path.basename(f_name), compression_cfg=(method, compression_cfgs[method]))
                    db[method][name] = torch.load(f_name, map_location=device, weights_only=False)
                db[method][name] = db[method][name].to(device)
    return db
        
def cal_norms(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> Dict[str, float]:
        """
        Calculate the norms between two tensors. Include l0, l1, l2, linf, and PSNR.

        Args:
            x (torch.Tensor): The first batch of images. Shape: [B, 3, H, W], range [0, 1], dtype torch.float32
            y (torch.Tensor): The second batch of images. Shape: [B, 3, H, W], range [0, 1], dtype torch.float32
            epsilon (float): The maximum perturbation value. 

        Returns:
            Dict[str, float]: A dictionary containing the norms.
        """
        resolution = x.size(2) * x.size(3)
        img_num = x.size(0)
        diffs = torch.abs(x - y)
        l0 = torch.sum(torch.any(diffs != 0, dim=1)).item() / (resolution * img_num)
        l1 = torch.sum(diffs).item() / (3*resolution*epsilon*img_num)
        l2 = torch.sum(torch.sqrt(diffs ** 2)).item() / img_num
        linf = torch.max(diffs).item()

        mse = torch.mean(diffs ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)).item() if mse > 0 else float('inf')
        return {'l0': l0, 'l1': l1, 'l2': l2, 'linf': linf, 'psnr': psnr}

def retrieve(db: torch.Tensor, db_labels: List[str], queries: torch.Tensor, query_labels: List[str], dist_func: str, topk: int, sorted_retrieval: bool = True, return_retrieved_idxs: bool = False) -> Union[List[float], Tuple[List[float], List[List[int]]]]:
    """
    Assume k queries in total. For each query, determine how many of the top-k retrievals are correct. 

    Args:
        db (torch.Tensor): The database of features. Shape: (N, D), where N is the number of features and D is the feature dimension.
        db_labels (List[str]): The labels of the database features.
        queries (torch.Tensor): The query features. Shape: (K, D), where K is the number of queries.
        query_labels (List[str]): The labels of the query features.
        dist_func (str): The distance function to use. Either 'cosine' or 'euclidean'.
        topk (int): The number of top retrievals to consider.
        sorted_retrieval (bool): Whether to sort the retrievals by the similarity/distance score. (Monotonically decreasing in terms of similarity, conceptually, whether cosine or euclidean distance)
        return_retrieved_idxs (bool): Whether to return the retrieved indexes in the db Tensor.

    Returns:
        Union[List[float], Tuple[List[float], List[List[int]]]]:
         - List[float]: For each query, how many of the top-k retrievals are correct. If return_retrieved_idxs is False.
         - Tuple[List[float], List[List[int]]]: For each query, how many of the top-k retrievals are correct, and the retrieved indexes in the db Tensor. If return_retrieved_idxs is True.
    """
    #k = queries.shape[0] # The number of queries
    k = topk
    if dist_func == 'cosine':
        matrix = torch.matmul(F.normalize(queries, p=2, dim=1), F.normalize(db, p=2, dim=1).T)
        db_matches_idxs = matrix.topk(k, dim=1, largest=True, sorted=sorted_retrieval)[1]
    elif dist_func == 'euclidean':
        matrix = torch.cdist(queries, db, p=2)
        db_matches_idxs = matrix.topk(k, dim=1, largest=False, sorted=sorted_retrieval)[1]
    else:
        raise ValueError(f"Unsupported distance function: {dist_func}. Use 'cosine' or 'euclidean'.")
    accus = []
    retrieved_idxs = []
    for i in range(queries.shape[0]):
        query_label = query_labels[i]
        db_matches = db_matches_idxs[i]
        correct_count = sum(1 for idx in db_matches if db_labels[idx] == query_label)
        accus.append(correct_count / k)
        if return_retrieved_idxs:
            retrieved_idxs.append(db_matches.detach().cpu().numpy().tolist())
    if return_retrieved_idxs:
        return accus, retrieved_idxs
    return accus

def prot_eval(orig_features: torch.Tensor, protected_features: torch.Tensor, face_db: Dict[str, torch.Tensor], dist_func: str, query_portion: float, device: torch.device, verbose: bool = False) -> Dict[str, float]:
    """
    Protection evaluation function. It evaluates the retrieval accuracy of protected and unprotected queries against a database of facial features.

    Args:
        orig_features (torch.Tensor): The original features of the images. Shape: (N, D), where N is the number of features and D is the feature dimension.
        protected_features (torch.Tensor): The protected features of the images. Shape: (N, D).
        face_db (Dict[str, torch.Tensor]): The database of facial features.
        dist_func (str): The distance function to use. Either 'cosine' or 'euclidean'.
        query_portion (float): The portion of the images to use as queries.
        device (torch.device): The device to use for computation.

    Returns:
        Dict[str, float]: A dictionary containing the retrieval accuracies for different cases. 
    """
    noise_features, noise_labels = [], []
    for name, features in face_db.items():
        noise_features.append(features)
        noise_labels.extend([name] * features.shape[0])
    noise_features = torch.cat(noise_features, dim=0)

    img_num = orig_features.shape[0]
    query_num = int(img_num * query_portion)
    query_labels = ['_user'] * query_num
    db_labels = noise_labels + ['_user'] * (img_num - query_num)
    noise_features, orig_features, protected_features = noise_features.to(device), orig_features.to(device), protected_features.to(device)

    dist_func = dist_func
    #dist_func = fr.dis_func
    # Case 1: Search with unprotected queries
    if verbose:
        print('===== Search with unprotected queries =====')    
    query_features = orig_features[:query_num]
    # (a) Protected entries should not be retrieved as top matches
    if verbose:
        print("Protected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, protected_features[query_num:]], dim=0)
    one_as = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func, 
                     topk=img_num - query_num)
    one_a = sum(one_as) / len(one_as)
    if verbose:
        print(f"1A Retrieval accuracy: {one_a:.4f} (lower is better)")
    # (b) Unprotected entries are deemed to be retrieved as top matches
    if verbose:
        print("Unprotected entries are deemed to be retrieved as top matches")
    db_features = torch.cat([noise_features, orig_features[query_num:]], dim=0)
    one_bs = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func, 
                     topk=img_num - query_num)
    one_b = sum(one_bs) / len(one_bs)
    if verbose:
        print(f"1B Retrieval accuracy: {one_b:.4f} (higher is better)")

    # Case 2: Search with protected queries
    if verbose:
        print('===== Search with protected queries =====')
    query_features = protected_features[:query_num]
    # (a) Protected entries should not be retrieved as top matches
    if verbose:
        print("Protected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, protected_features[query_num:]], dim=0)
    two_as = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func, 
                     topk=img_num - query_num)
    two_a = sum(two_as) / len(two_as)
    if verbose:
        print(f"2A Retrieval accuracy: {two_a:.4f} (lower is better)")
    # (b) Unprotected entries should not be retrieved as top matches
    if verbose:
        print("Unprotected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, orig_features[query_num:]], dim=0)
    two_bs = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func, 
                     topk=img_num - query_num)
    two_b = sum(two_bs) / len(two_bs)
    if verbose:
        print(f"2B Retrieval accuracy: {two_b:.4f} (lower is better)")

    return {'1a': one_a,'1b': one_b,'2a': two_a,'2b': two_b}

def eval_masks(three_d: bool, 
            test_data: DataLoader, 
            face_db: Dict[str, torch.Tensor], 
            fr: FR, 
            device: torch.device, 
            bin_mask: bool, 
            epsilon: float, 
            masks: torch.Tensor,
            query_portion: float = 0.5, 
            vis_eval: bool = True, 
            lpips_backbone: str = "vgg", 
            verbose: bool = False, 
            replication_attack: bool = False, 
            uvmapper: Optional[UVGenerator] = None, 
            replication_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Evaluate masks against a database of facial features.

    Args:
        three_d (bool): Whether to use 3D masks.
        test_data (DataLoader): The DataLoader containing the test data.
        face_db (Dict[str, torch.Tensor]): The database of facial features.
        fr (FR): The facial recognition model.
        device (torch.device): The device to use for computation.
        bin_mask (bool): Whether to use binary masks.
        epsilon (float): The maximum perturbation value.
        masks (torch.Tensor): The masks to apply. Shape: (B, 3, H, W). Range: [-epsilon, epsilon]. dtype: np.float32.
        query_portion (float): The portion of the test data to use as queries.
        vis_eval (bool): Whether to visualize the evaluation results.
        lpips_backbone (str): The backbone to use for LPIPS calculation. 'vgg', 'alex', or 'squeeze'.
        verbose (bool): Whether to print the evaluation results.
        replication_attack (bool): Whether to perform replication attack evaluation.
        uvmapper (Optional[UVGenerator]): The UV mapper to use for 3D masks
        rep_mask (Optional[torch.Tensor]): The replication attack mask. Shape: (B, 3, H, W). Range: [-epsilon, epsilon]. dtype: np.float32.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation results. Includes retrieval accuracies, various norms, and visual quality metrics.
    """
    tensor_masks = masks
    img_cnt = 0
    orig_features, protected_features = [], []
    ssims, psnrs, lpipses, l0s, l1s, l2s, linfs = [], [], [], [], [], [], []
    if vis_eval:
        lpips_evaluator = lpips.LPIPS(net=lpips_backbone).to(device)
    for idx, tensors in tqdm.tqdm(enumerate(test_data), total=len(test_data), desc="Applying mask, extracting features, and evaluating visual quality"):
        orig_faces = tensors[0].to(device)  # Shape: [B, 3, H, W]
        img_num = orig_faces.shape[0]
        if three_d:
            uvs = tensors[1].to(device)
            textures = tensor_masks[img_cnt:img_cnt + img_num].to(device)  # Shape: [B, 224, 224, 3]
            perturbations = torch.clamp(F.grid_sample(textures, uvs, mode='bilinear', align_corners=True), -epsilon, epsilon)
        else:
            perturbations = tensor_masks[img_cnt:img_cnt + img_num].to(device)
        if bin_mask:
            bin_masks = tensors[2].to(device)
            perturbations *= bin_masks
        protected_faces = torch.clamp(orig_faces + perturbations, 0, 1)
        if replication_attack:
            uvs, bin_masks, _ = uvmapper.forward(protected_faces)
            rep_perts = torch.clamp(F.grid_sample(replication_mask.repeat(protected_faces.shape[0], 1, 1, 1), uvs, mode='bilinear', align_corners=True), -epsilon, epsilon) if three_d else replication_mask.repeat(protected_faces.shape[0], 1, 1, 1)
            if bin_mask:
                rep_perts = rep_perts * bin_masks
            protected_faces = torch.clamp(protected_faces - rep_perts, 0, 1)
        img_cnt += img_num
        orig_features.append(fr(orig_faces).cpu())
        protected_features.append(fr(protected_faces).cpu())
        for prot_img, orig_img in zip(protected_faces, orig_faces):
            prot_img = prot_img.unsqueeze(0)
            orig_img = orig_img.unsqueeze(0)
            if vis_eval:
                """res = lpips_evaluator(prot_img * 2 - 1, orig_img * 2 - 1)
                lpips_value = res.mean().detach().cpu().numpy().item()
                res_min, res_max = res.min(), res.max()
                heat_map = (res - res.min()) / (res.max() - res.min() + 1e-10)
                heat_map = heat_map[0, 0].cpu().numpy()
                orig_np = (orig_img.clone().squeeze(0).permute(1, 2, 0).cpu().numpy())
                prot_np = (prot_img.clone().squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1); plt.imshow(orig_np); plt.title('Original Image'); plt.axis('off')
                plt.subplot(1, 3, 2); plt.imshow(prot_np); plt.title('Protected Image'); plt.axis('off')
                plt.subplot(1, 3, 3); plt.imshow(np.clip(orig_np, 0, 1)); plt.imshow(heat_map, cmap='magma', alpha=0.5); plt.title(f'LPIPS Heat Map (min: {res_min:.4f}, max: {res_max:.4f})'); plt.axis('off')
                plt.suptitle(f'LPIPS: {lpips_value:.4f}')
                plt.savefig(f"/home/zlwang/ProtegoPlus/trash/heat_maps/spatial_lpipsfalse_{idx}.png")
                lpipses.append(lpips_value)"""
                lpipses.append(lpips_evaluator(prot_img * 2 - 1, orig_img * 2 - 1).mean().detach().cpu().numpy().item())
                ssims.append(calc_ssim(prot_img, orig_img, data_range=1, size_average=True).cpu().numpy().item())
                norms = cal_norms(prot_img, orig_img, epsilon)
                l0s.append(norms['l0'])
                l1s.append(norms['l1'])
                l2s.append(norms['l2'])
                linfs.append(norms['linf'])
                psnrs.append(norms['psnr'])
    del orig_faces, protected_faces, perturbations
    complete_del()

    if not vis_eval:
        psnrs, ssims, lpipses, l0s, l1s, l2s, linfs = [0], [0], [0], [0], [0], [0], [0]
    mean_l0 = np.mean(l0s).item()
    max_l0 = max(l0s)
    min_l0 = min(l0s)
    mean_l1 = np.mean(l1s).item()
    max_l1 = max(l1s)
    min_l1 = min(l1s)
    mean_l2 = np.mean(l2s).item()
    max_l2 = max(l2s)
    min_l2 = min(l2s)
    mean_linf = np.mean(linfs).item() * 255
    max_linf = max(linfs) * 255
    min_linf = min(linfs) * 255
    mean_ssim = np.mean(ssims).item()
    max_ssim = max(ssims)
    min_ssim = min(ssims)
    mean_psnr = np.mean(psnrs).item()
    max_psnr = max(psnrs)
    min_psnr = min(psnrs)
    mean_lpips = np.mean(lpipses).item()
    max_lpips = max(lpipses)
    min_lpips = min(lpipses)
    orig_features = torch.cat(orig_features, dim=0).to(device)
    protected_features = torch.cat(protected_features, dim=0).to(device)

    prot_res = prot_eval(orig_features=orig_features,
                         protected_features=protected_features,
                         face_db=face_db,
                         dist_func='cosine',
                         query_portion=query_portion,
                         device=device,
                         verbose=verbose)
    one_a = prot_res['1a']
    one_b = prot_res['1b']
    two_a = prot_res['2a']
    two_b = prot_res['2b']

    return {
        '1a': one_a,
        '1b': one_b,
        '2a': two_a,
        '2b': two_b,
        'l0': mean_l0,
        'l0_max': max_l0,
        'l0_min': min_l0,
        'l1': mean_l1,
        'l1_max': max_l1,
        'l1_min': min_l1,
        'l2': mean_l2,
        'l2_max': max_l2,
        'l2_min': min_l2,
        'linf': mean_linf,
        'linf_max': max_linf,
        'linf_min': min_linf,
        'ssim': mean_ssim,
        'ssim_max': max_ssim,
        'ssim_min': min_ssim,
        'psnr': mean_psnr,
        'psnr_max': max_psnr,
        'psnr_min': min_psnr, 
        'lpips': mean_lpips,
        'lpips_max': max_lpips,
        'lpips_min': min_lpips
    }

def eval_mask_end2end(three_d: bool, 
                    test_raw_imgs: List[torch.Tensor],
                    face_db_path: str, 
                    frs: List[FR], 
                    fd: FD, 
                    uvmapper: UVGenerator, 
                    device: torch.device, 
                    bin_mask: bool, 
                    epsilon: float, 
                    mask: torch.Tensor,
                    query_portion: float = 0.5, 
                    strict_crop: bool = False, 
                    resize_face: bool = True, 
                    jpeg: bool = False, 
                    smoothing: str = None, 
                    vis_eval: bool = True, 
                    lpips_backbone: str = "vgg", 
                    verbose: bool = False, 
                    eval_scene1_db: Optional[Dict[str, Dict[str, torch.Tensor]]] = None, 
                    replication_attack: bool = False, 
                    replication_mask: Optional[torch.Tensor] = None) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate masks against a database of facial features in an end-to-end manner, aka starting from uncropped images to protection evaluation.

    Args:
        three_d (bool): Whether to use 3D masks.
        test_raw_imgs (List[torch.Tensor]): The list of original uncropped images. Each tensor has shape (1, 3, H, W), range [0, 1], dtype: torch.float32.
        face_db_path (str): The path to the database of facial features.
        frs (List[FR]): The list of facial recognition models.
        fd (FD): The face detection model.
        uvmapper (UVGenerator): The UV mapper for generating UV maps.
        device (torch.device): The device to use for computation.
        bin_mask (bool): Whether to use binary masks.
        epsilon (float): The maximum perturbation value.
        mask (torch.Tensor): The mask to apply. Shape: (1, 3, H, W). Range: [-epsilon, epsilon]. dtype: np.float32.
        resize_face (bool): Whether to resize the face or the mask.
        jpeg (bool): Whether to apply JPEG compression with quality 75 to the protected images before feature extraction.
        query_portion (float): The portion of the test data to use as queries.
        strict_crop (bool): The strict mode that will be passed to protego.utils.crop_face function.
        vis_eval (bool): Whether to visualize the evaluation results.
        lpips_backbone (str): The backbone to use for LPIPS calculation. 'vgg', 'alex', or 'squeeze'.
        verbose (bool): Whether to print the evaluation results.
        eval_scene1_db (Optional[Dict[str, Dict[str, torch.Tensor]]]): The pre-built database for scene 1 evaluation. If None, it will be built solely from face_db_path.
        replication_attack (bool): Whether to perform a replication attack evaluation.
        replication_masks (Optional[torch.Tensor]): The masks to use for replication attack evaluation. Shape: (N, 3, H, W). Range: [-epsilon, epsilon]. dtype: torch.float32.

    Returns:
        Dict[str, Union[float, Dict[str, float]]]: A dictionary containing the evaluation results. Includes retrieval accuracies for each FR model, various norms, and visual quality metrics.
    """
    resized_faces, faces, positions = [], [], []
    for img_idx, raw_img in tqdm.tqdm(enumerate(test_raw_imgs), desc="Detecting and cropping faces from raw images"):
        raw_img = raw_img.to(device).squeeze(0)
        #print(raw_img.shape, raw_img.min(), raw_img.max())
        face, pos = crop_face(raw_img, fd, verbose=False, strict=strict_crop)
        if face is None or pos is None:
            print(f"Warning: No valid face detected in image {img_idx} (strict = {strict_crop}). Skipping.")
            continue
        #print(face.shape, face.min(), face.max(), face.mean())
        #face.div_(255.)
        pos = list(pos)
        pos.append(img_idx)
        resized_faces.append(F.interpolate(face.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0))
        if not resize_face:
            faces.append(face)
        positions.append(pos)
    resized_faces = torch.stack(resized_faces, dim=0)
    uvs, bin_masks, _ = uvmapper.forward(imgs=resized_faces)
    perts = torch.clamp(F.grid_sample(mask.repeat(resized_faces.shape[0], 1, 1, 1), uvs, mode='bilinear', align_corners=True), -epsilon, epsilon) if three_d else mask.repeat(faces.shape[0], 1, 1, 1)
    if bin_mask:
        perts = perts * bin_masks
    protected_imgs = []
    for pos_idx, pos in enumerate(positions):
        x1, y1, x2, y2, img_idx = pos
        protected_img = test_raw_imgs[img_idx].clone().squeeze(0)
        #print(protected_img.shape, protected_img.min(), protected_img.max())
        if resize_face:
            protected_face = torch.clamp(resized_faces[pos_idx] + perts[pos_idx], 0, 1)
            protected_face = F.interpolate(protected_face.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
            protected_img[:, y1:y2, x1:x2] = protected_face
        else:
            pert = F.interpolate(perts[[pos_idx]], size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
            #print(pert.max(), pert.min())
            protected_face = torch.clamp(faces[pos_idx] + pert, 0, 1)
            protected_img[:, y1:y2, x1:x2] = protected_face
        protected_img = protected_img.unsqueeze(0)
        if jpeg:
            protected_img = compress(protected_img, method='jpeg', quality=75)
        protected_img = protected_img.mul(255.).to(torch.uint8).float().div(255.)
        if smoothing is not None:
            if smoothing == 'gaussian':
                protected_img = compress(protected_img, method='gaussian', kernel_size = 3, sigma=0.7)
            elif smoothing == 'median':
                protected_img = compress(protected_img, method='median', kernel_size = 3)
            else:
                raise ValueError(f"Unsupported smoothing method: {smoothing}. Use 'gaussian', 'median', or None.")
        protected_imgs.append(protected_img.squeeze(0))
    del faces, perts, uvs, bin_masks
    complete_del()
    #protected_imgs = torch.stack(protected_imgs, dim=0)
    protected_faces = []
    have_face = []
    for img_idx, protected_img in tqdm.tqdm(enumerate(protected_imgs), desc="Re-detecting and cropping faces from protected images"):
        #print(protected_img.shape, protected_img.min(), protected_img.max())
        # Visualization for debugging
        #_protected_img_vis = (protected_img.detach().permute(1, 2, 0).mul(255.).cpu().numpy().astype(np.uint8))
        #cv2.imwrite(f"/home/zlwang/ProtegoPlus/trash/protected_img/protected_img_{img_idx}.png", cv2.cvtColor(_protected_img_vis, cv2.COLOR_RGB2BGR))
        protected_face, pos = crop_face(protected_img, fd, verbose=False, strict=strict_crop)
        if protected_face is None or pos is None:
            print(f"Warning: No valid face detected in protected image {img_idx} (strict = {strict_crop}). Skipping.")
            continue
        #protected_face.div_(255.)
        protected_faces.append(F.interpolate(protected_face.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0))
        have_face.append(img_idx)
    protected_faces = torch.stack(protected_faces, dim=0)
    # Replication Attack
    if replication_attack and replication_mask is not None:
        uvs, bin_masks, _ = uvmapper.forward(imgs=protected_faces)
        rep_perts = torch.clamp(F.grid_sample(replication_mask.repeat(protected_faces.shape[0], 1, 1, 1), uvs, mode='bilinear', align_corners=True), -epsilon, epsilon) if three_d else replication_mask.repeat(protected_faces.shape[0], 1, 1, 1)
        if bin_mask:
            rep_perts = rep_perts * bin_masks
        protected_faces = torch.clamp(protected_faces - rep_perts, 0, 1)
    original_faces = resized_faces[have_face]
    l0s, l1s, l2s, linfs, psnrs, ssims, lpipses = [0], [0], [0], [0], [0], [0], [0]
    if vis_eval:
        dl = DataLoader(dataset=TensorDataset(original_faces, protected_faces), batch_size=16, shuffle=False, num_workers=4)
        lpips_evaluator = lpips.LPIPS(net=lpips_backbone).to(device)
        for tensors in dl:
            orig_faces, prot_faces = tensors
            orig_faces, prot_faces = orig_faces.to(device), prot_faces.to(device)
            norms = cal_norms(prot_faces, orig_faces, epsilon)
            l0s.append(norms['l0'])
            l1s.append(norms['l1'])
            l2s.append(norms['l2'])
            linfs.append(norms['linf'])
            psnrs.append(norms['psnr'])
            ssims.append(calc_ssim(prot_faces, orig_faces, data_range=1, size_average=True).detach().cpu().numpy().item())
            lpipses.append(lpips_evaluator(prot_faces * 2 - 1, orig_faces * 2 - 1).mean().detach().cpu().numpy().item())
    mean_l0 = np.mean(l0s).item()
    max_l0 = max(l0s)
    min_l0 = min(l0s)
    mean_l1 = np.mean(l1s).item()
    max_l1 = max(l1s)
    min_l1 = min(l1s)
    mean_l2 = np.mean(l2s).item()
    max_l2 = max(l2s)
    min_l2 = min(l2s)
    mean_linf = np.mean(linfs).item() * 255
    max_linf = max(linfs) * 255
    min_linf = min(linfs) * 255
    mean_ssim = np.mean(ssims).item()
    max_ssim = max(ssims)
    min_ssim = min(ssims)
    mean_psnr = np.mean(psnrs).item()
    max_psnr = max(psnrs)
    min_psnr = min(psnrs)
    mean_lpips = np.mean(lpipses).item()
    max_lpips = max(lpipses)
    min_lpips = min(lpipses)
    fr_results = {}
    del l0s, l1s, l2s, linfs, psnrs, ssims, lpipses
    complete_del()
    for fr in frs:
        protected_features = fr(protected_faces)
        orig_features = fr(original_faces)
        noise_db = build_facedb(face_db_path, fr.model_name, device)
        if eval_scene1_db is not None:
            noise_db.update(eval_scene1_db[fr.model_name])
        fr_results[fr.model_name] = prot_eval(orig_features=orig_features,
                                             protected_features=protected_features,
                                             face_db=noise_db,
                                             dist_func='cosine',
                                             query_portion=query_portion,
                                             device=device,
                                             verbose=verbose)
    return {
        'prot_results': fr_results,
        'l0': mean_l0,
        'l0_max': max_l0,
        'l0_min': min_l0,
        'l1': mean_l1,
        'l1_max': max_l1,
        'l1_min': min_l1,
        'l2': mean_l2,
        'l2_max': max_l2,
        'l2_min': min_l2,
        'linf': mean_linf,
        'linf_max': max_linf,
        'linf_min': min_linf,
        'ssim': mean_ssim,
        'ssim_max': max_ssim,
        'ssim_min': min_ssim,
        'psnr': mean_psnr,
        'psnr_max': max_psnr,
        'psnr_min': min_psnr, 
        'lpips': mean_lpips,
        'lpips_max': max_lpips,
        'lpips_min': min_lpips
    }

def compression_eval(compression_methods: List[str], 
                    compression_cfgs: Dict[str, Dict[str, Any]],
                    three_d: bool, 
                    test_data: DataLoader, 
                    face_db: Dict[str, Dict[str, torch.Tensor]], 
                    fr: FR, 
                    device: torch.device, 
                    bin_mask: bool, 
                    epsilon: float, 
                    masks: torch.Tensor, 
                    query_portion: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the masks' robustness against different compression methods.

    Args:
        compression_methods (List[str]): The list of compression methods to evaluate.
        compression_cfgs (Dict[str, Dict[str, Any]]): The configurations for each compression method.
        three_d (bool): Whether to use 3D masks.
        test_data (DataLoader): The DataLoader containing the test data.
        face_db (Dict[str, Dict[str, torch.Tensor]]): The database of facial features for different compression methods.
        fr (FR): The facial recognition model.
        device (torch.device): The device to use for computation.
        bin_mask (bool): Whether to use binary masks.
        epsilon (float): The maximum perturbation value.
        masks (torch.Tensor): The masks to apply. Shape: (B, 3, H, W). Range: [-epsilon, epsilon]. dtype: torch.float32.
        query_portion (float): The portion of the test data to use as queries.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the evaluation results for each compression method.
        Each method's results include '1a', '1b', '2a', '2b'.
    """

    tensor_masks = masks.clone()
    prot_res = {}
    for method in compression_methods:
        img_cnt = 0
        orig_features, protected_features = [], []
        for idx, tensors in tqdm.tqdm(enumerate(test_data), total=len(test_data), desc=f"Compressing imgs with {method}, Applying mask, and extracting features"):
            orig_faces = tensors[0].to(device)  # Shape: [B, 3, H, W]
            img_num = orig_faces.shape[0]
            if three_d:
                uvs = tensors[1].to(device)
                textures = tensor_masks[img_cnt:img_cnt + img_num].to(device)  # Shape: [B, 224, 224, 3]
                perturbations = torch.clamp(F.grid_sample(textures, uvs, mode='bilinear', align_corners=True), -epsilon, epsilon)
            else:
                perturbations = tensor_masks[img_cnt:img_cnt + img_num].to(device)
            if bin_mask:
                bin_masks = tensors[2].to(device)
                perturbations *= bin_masks
            protected_faces = torch.clamp(orig_faces + perturbations, 0, 1)
            compressed_protected_faces = compress(imgs = protected_faces, method = method, differentiable = False, **compression_cfgs[method])
            compressed_orig_faces = compress(imgs = orig_faces, method = method, differentiable = False, **compression_cfgs[method])

            img_cnt += img_num
            orig_features.append(fr(compressed_orig_faces).cpu())
            protected_features.append(fr(compressed_protected_faces).cpu())
            """for prot_img, orig_img in zip(protected_faces, orig_faces):
                prot_img = prot_img.unsqueeze(0)
                orig_img = orig_img.unsqueeze(0)"""
        del orig_faces, protected_faces, perturbations
        complete_del()
        orig_features = torch.cat(orig_features, dim=0).to(device)
        protected_features = torch.cat(protected_features, dim=0).to(device)

        prot_res[method] = prot_eval(orig_features=orig_features,
                                    protected_features=protected_features,
                                    face_db=face_db[method],
                                    dist_func='cosine',
                                    query_portion=query_portion,
                                    device=device)
    return prot_res

def visualize_mask(
    orig_img: torch.Tensor,
    uv: torch.Tensor,
    bin_mask: torch.Tensor,
    univ_mask: torch.Tensor,
    save_path: str,
    epsilon: float, 
    use_bin_mask: bool,
    three_d: bool
) -> None:
    """
    Visualize the original image, protected image, universal mask, perturbation, UV map, and binary mask.

    Args:
        orig_img (torch.Tensor): The original image. Shape: (3, H, W). Range: [0, 1].
        uv (torch.Tensor): The UV map. Shape: (H, W, 2). 
        bin_mask (torch.Tensor): The binary mask. Shape: (1, H, W).
        univ_mask (torch.Tensor): The universal mask. Shape: (1, 3, H, W). Range: [-epsilon, epsilon].
        save_path (str): The path to save the visualization.
        epsilon (float): The maximum perturbation value.
        use_bin_mask (bool): Whether to use the binary mask.
        three_d (bool): Whether to use 3D masks.
    """
    _orig_img = orig_img.detach()
    _uv = uv.detach()
    _bin_mask = bin_mask.detach()
    _univ_mask = univ_mask.detach()
    if three_d:
        _pert = torch.clamp(F.grid_sample(_univ_mask, _uv.unsqueeze(0), mode='bilinear', align_corners=True).squeeze(0), -epsilon, epsilon)
    else:
        _pert = _univ_mask.squeeze(0)
    if use_bin_mask:
        _pert = _pert * _bin_mask
    _prot_img = torch.clamp(_orig_img + _pert, 0, 1)

    _orig_img = _orig_img.permute(1, 2, 0).mul(255.).cpu().contiguous().numpy().astype(np.uint8)
    _prot_img = _prot_img.permute(1, 2, 0).mul(255.).cpu().contiguous().numpy().astype(np.uint8)
    _pert = (_pert - _pert.min()) / (_pert.max() - _pert.min())
    if use_bin_mask:
        _pert = _pert * _bin_mask
    _pert = _pert.detach().permute(1, 2, 0).mul(255.).cpu().contiguous().numpy().astype(np.uint8)
    _univ_mask = (_univ_mask - _univ_mask.min()) / (_univ_mask.max() - _univ_mask.min())
    _univ_mask = _univ_mask.squeeze(0).permute(1, 2, 0).mul(255.).cpu().contiguous().numpy().astype(np.uint8)

    _uv = (_uv - _uv.min()) / (_uv.max() - _uv.min())
    _uv = _uv.mul(255).cpu().contiguous().numpy().astype(np.uint8)
    _uv_img = np.zeros((_uv.shape[0], _uv.shape[1], 3), dtype=np.uint8)
    _uv_img[..., 1:] = _uv
    _uv = _uv_img
    # Turn binary mask into a black-white image
    _bin_mask = _bin_mask.mul(255).permute(1, 2, 0).cpu().contiguous().numpy().astype(np.uint8)
    _bin_mask = cv2.cvtColor(_bin_mask, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(12, 8))
    for i, (_img, _title) in enumerate(zip([_orig_img, _prot_img, _univ_mask, _pert, _uv, _bin_mask], 
                                        ['Original Image', 'Protected Image', 'Universal Mask', 'Perturbation', f'UV({"Used" if three_d else "Unused"})', f'Binary Mask({"Used" if use_bin_mask else "Unused"})'])):
        plt.subplot(2, 3, i+1)
        plt.imshow(_img)
        plt.title(_title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_augmented_ims(orig_imgs: torch.Tensor, methods: List[str], aug_cfgs: Dict[str, Any]) -> torch.Tensor:
    augmented_imgs = orig_imgs.clone()
    for method in methods:
        augmented_imgs = compress(imgs=augmented_imgs, method=method, differentiable=False, **aug_cfgs[method])
    return augmented_imgs