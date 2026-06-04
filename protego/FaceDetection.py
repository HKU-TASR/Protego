import os
from typing import List, Dict, Tuple, Optional, Any
from math import ceil
from itertools import product as product
import zipfile

import torch
from torchvision.ops.boxes import nms
from torchvision import transforms
import gdown

from FD_DB.Retinaface.net import *
from FD_DB.MTCNN.mtcnn import MTCNN
from . import BASE_PATH

FD_DB_PATH = os.path.join(BASE_PATH, 'FD_DB')
RETINAFACE_HOME = os.path.join(FD_DB_PATH, 'Retinaface')
MTCNN_HOME = os.path.join(FD_DB_PATH, 'MTCNN')

LANDMARK_POOL = ['mobilenet_retinaface_widerface', 'resnet50_retinaface_widerface', 'mtcnn']
DETECTION_ONLY_POOL = []

def download(paths: List[str], urls: List[str]) -> None:
    """
    A crude download util function, only tailoreed for the scenarios this project meets. 

    Args:
        path (str): The local desired path of the weight file
        url (str): The URL of the file to download.
    """
    for path, url in zip(paths, urls):
        folder = "/".join(path.split("/")[:-1])+'/'
        if not os.path.exists(path):
            print(f"Weight {path} does not exist. Downloading it from {url}")
            if "uc?id" in url:
                file_id = url.split("uc?id=")[-1]
                downloaded_f = gdown.download(id=file_id, output=folder, quiet=False)
            elif "view?usp=sharing" in url or "view?usp=share_link" in url:
                downloaded_f = gdown.download(url=url, output=folder, fuzzy=True, quiet=False)
            if downloaded_f.endswith(".zip"):
                with zipfile.ZipFile(downloaded_f, 'r') as zip_ref:
                    zip_ref.extractall(folder)
                os.remove(downloaded_f)
            elif downloaded_f != path:
                os.rename(downloaded_f, path)
    return

@torch.no_grad()
class FD(object):
    def __init__(self, model_name: str = "mobilenet_retinaface_widerface", device: torch.device = torch.device("cpu")):
        self.device = device
        self.model_name = model_name
        
        if self.model_name == 'mobilenet_retinaface_widerface':
            self.fd_model = Retinaface(arch='mobilenet0.25', device=device)
        elif self.model_name == 'resnet50_retinaface_widerface':
            self.fd_model = Retinaface(arch='resnet50', device=device)
        elif self.model_name == 'mtcnn':
            self.fd_model = MTCNN_Wrapper(device=device)
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}.")

        self.drange = self.fd_model.drange
        self.ldmk_num = self.fd_model.ldmk_num

    def __call__(self, img: torch.Tensor, conf_thresh: Any = None, nms_thresh: Any = None) -> Optional[List[Tuple[int, int, int, int, float, List[int]]]]:
        """
        Combine preprocessing and forward pass for convenience.

        Args:
            img (torch.Tensor): Input image tensor of shape [3, H, W], range [0, 255] or [0, 1], RGB, dtype torch.float32, on the same device as the model.
        
        Returns:
            Optional[List[Tuple[int, int, int, int, float, [List[int]]]]: A list of detected faces with bounding boxes, confidence scores, and landmark points.
                Each element is a tuple (x1, y1, x2, y2, score, landmarks).
               landmark length depends on the model, will be empty list if the model does not provide landmarks.
                Returns None if no faces are detected.
        """
        return self.forward(self.preprocess(img), conf_thresh=conf_thresh, nms_thresh=nms_thresh)

    def forward(self, img: torch.Tensor, conf_thresh: Any = None, nms_thresh: Any = None) -> Optional[List[Tuple[int, int, int, int, float, List[int]]]]:
        """
        Args:
            img (torch.Tensor): Preprocessed input image tensor matching the model's expected input format.

        Returns:
            Optional[List[Tuple[int, int, int, int, float, [List[int]]]]: A list of detected faces with bounding boxes, confidence scores, and landmark points.
                Each element is a tuple (x1, y1, x2, y2, score, landmarks).
               landmark length depends on the model, will be empty list if the model does not provide landmarks.
                Returns None if no faces are detected.
        """
        return self.fd_model.forward(img, conf_thresh=conf_thresh, nms_thresh=nms_thresh)

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input image tensor to match the model's expected input format.

        Args:
            img (torch.Tensor): Input image tensor of shape [3, H, W], range [0, 255] or [0, 1], RGB, dtype torch.float32, on the same device as the model.
        """
        if img.max() <= 2 and self.drange == 255:
            rescaled_img = img.float() * 255.
        elif img.max() <= 2 and self.drange == 1:
            rescaled_img = img.float()
        elif img.max() > 2 and self.drange == 255:
            rescaled_img = img.float()
        elif img.max() > 2 and self.drange == 1:
            rescaled_img = img.float() / 255.
        preprocessed = self.fd_model.preprocessing(rescaled_img).unsqueeze(0)
        return preprocessed
    
class MTCNN_Wrapper(object):
    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.ldmk_num = 5
        self.path_dict = {
            'mtcnn': [os.path.join(MTCNN_HOME, 'pretrained/pnet.npy'),
                      os.path.join(MTCNN_HOME, 'pretrained/rnet.npy'),
                      os.path.join(MTCNN_HOME, 'pretrained/onet.npy')]
        }
        self.url_dict = {
            'mtcnn': ["https://drive.google.com/file/d/1uJopXpkHHzzImZ-4LVWrRHHMbUECi5Fb/view?usp=share_link",
                      "https://drive.google.com/file/d/1uJopXpkHHzzImZ-4LVWrRHHMbUECi5Fb/view?usp=share_link",
                      "https://drive.google.com/file/d/1uJopXpkHHzzImZ-4LVWrRHHMbUECi5Fb/view?usp=share_link"]
        }
        for key in self.path_dict:
            download(self.path_dict[key], self.url_dict[key])
        self.model = MTCNN(weight_paths={'pnet': self.path_dict['mtcnn'][0],
                                         'rnet': self.path_dict['mtcnn'][1],
                                         'onet': self.path_dict['mtcnn'][2]}, device=device)
        self.preprocessing = transforms.Compose([transforms.Lambda(lambda x: x)]) # Identity transform
        self.drange = 255

    def forward(self, img: torch.Tensor, conf_thresh: Any = None, nms_thresh: Any = None) -> Optional[List[Tuple[int, int, int, int, float, List[int]]]]:
        conf_threshes = conf_thresh if conf_thresh is not None else [0.6, 0.7, 0.8]
        nms_threshes = nms_thresh if nms_thresh is not None else [0.7, 0.7, 0.7]
        bboxs, ldmks = self.model.detect(img, return_zero=False, return_type='list', conf_threshes=conf_threshes, nms_threshes=nms_threshes)
        #print(bboxs, ldmks)
        if bboxs is None or ldmks is None:
            return None
        bboxs, ldmks = bboxs[0], ldmks[0]  # input is a single image
        dets = []
        for bbox, ldmk in zip(bboxs, ldmks):
            bbox = bbox.cpu().numpy()
            ldmk = ldmk.cpu().numpy()
            #print(bbox.shape, ldmk.shape)
            x1, y1, x2, y2, score = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            ldmk = [x for x in ldmk]
            dets.append((x1, y1, x2, y2, score, ldmk))
        return dets

class Retinaface(object):
    def __init__(self, arch: str, device: torch.device = torch.device("cpu")) -> None:
        """
        Model: RetinaFace
        Dataset: Widerface
        Source: https://github.com/biubug6/Pytorch_Retinaface

        Args:
            arch (str): Model architecture. Options: 'mobilenet0.25', 'resnet50'
            device (torch.device): Device to run the model on.
        """
        self.device = device
        self.ldmk_num = 5  # Number of landmark points (5 points: left eye, right eye, nose, left mouth corner, right mouth corner)
        self.path_dict = {
            'mobilenet0.25': [os.path.join(RETINAFACE_HOME, 'pretrained/mobilenet0.25_Final.pth'), 
                              os.path.join(RETINAFACE_HOME, 'pretrained/mobilenetV1X0.25_pretrain.tar')],
            'resnet50': [os.path.join(RETINAFACE_HOME, 'pretrained/Resnet50_Final.pth')]
        }
        self.url_dict = {
            'mobilenet0.25': ["https://drive.google.com/file/d/1AlY3yMYFGm0dTi7-lN4bq6T2McEJqrsv/view?usp=sharing", 
                              "https://drive.google.com/file/d/1q36RaTZnpHVl4vRuNypoEMVWiiwCqhuD/view?usp=share_link"], 
            'resnet50': ["https://drive.google.com/file/d/1RvJEqJL1htXEQ6tb9noMCq7v9tOgag_0/view?usp=sharing"]
        }
        download(self.path_dict[arch], self.url_dict[arch])
        if arch == 'mobilenet0.25':
            self.cfg = {
            'name': 'mobilenet0.25',
            'statedict_path': os.path.join(RETINAFACE_HOME, 'pretrained/mobilenetV1X0.25_pretrain.tar'),
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 32,
            'ngpu': 1,
            'epoch': 250,
            'decay1': 190,
            'decay2': 220,
            'image_size': 640,
            'pretrain': True,
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32,
            'out_channel': 64
            }
        elif arch == 'resnet50':
            self.cfg = {
            'name': 'Resnet50',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 24,
            'ngpu': 4,
            'epoch': 100,
            'decay1': 70,
            'decay2': 90,
            'image_size': 840,
            'pretrain': True,
            'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
            'in_channel': 256,
            'out_channel': 256
            }
        self.model = RetinaFace(cfg=self.cfg, phase='test')
        state_dict = torch.load(self.path_dict[arch][0], map_location=device, weights_only=False)
        if 'state_dict' in state_dict.keys():
            state_dict = self.remove_prefix(state_dict['state_dict'], 'module.')
        else:
            state_dict = self.remove_prefix(state_dict, 'module.')
        self.check_keys(self.model, state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device).eval()

        self.preprocessing = transforms.Compose([
            transforms.Lambda(lambda x: x[[2, 1, 0], :, :]),
            transforms.Lambda(lambda x: x - x.new_tensor([104., 117., 123.])[:, None, None]), 
        ])
        self.drange = 255

    def forward(self, img: torch.Tensor, conf_thresh: float = 0.8, nms_thresh: float = 0.4) -> Optional[List[Tuple[int, int, int, int, float, List[int]]]]:
        """
        Args:
            img (torch.Tensor): Input image tensor of shape [1, 3, H, W], range [-128, 127], RGB, dtype torch.float32
            conf_thresh (float): Confidence threshold for face detection.
            nms_thresh (float): Non-maximum suppression threshold.

        Returns:
            Optional[List[Tuple[int, int, int, int, float, List[int]]]]: A list of detected faces with bounding boxes, confidence scores, and landmark points.
                Each element is a tuple (x1, y1, x2, y2, score, landmarks).
                landmarks is a list of 10 integers representing 5 (x, y) points.
                Returns None if no faces are detected.
        """
        conf_thresh = conf_thresh if conf_thresh is not None else 0.8
        nms_thresh = nms_thresh if nms_thresh is not None else 0.4
        _, _, H, W = img.shape
        scale_bbox = torch.Tensor([W, H, W, H]).to(self.device)
        loc, conf, ldmks = self.model(img) # loc: [1, num_priors, 4], conf: [1, num_priors, 2], ldmks: [1, num_priors, 10]
        prior_data = self.prior_box((H, W), self.cfg, self.device).data
        boxes = self.decode_bbox(loc.data.squeeze(0), prior_data, self.cfg['variance']) # shape [num_priors, 4]
        boxes = boxes * scale_bbox # back to original scale
        conf = conf.squeeze(0).data[:, 1] # shape [num_priors]
        ldmks = self.decode_ldmk(ldmks.data.squeeze(0), prior_data, self.cfg['variance']) # shape [num_priors, 10]
        scale_ldmks = torch.Tensor([W, H, W, H, W, H, W, H, W, H]).to(self.device)
        ldmks = ldmks * scale_ldmks # back to original scale
        inds = torch.where(conf > conf_thresh)[0]
        boxes = boxes[inds] # shape [num_filtered, 4]
        conf = conf[inds] # shape [num_filtered]
        ldmks = ldmks[inds] # shape [num_filtered, 10]
        if boxes.shape[0] == 0:
            return None
        ordered_inds = torch.argsort(conf, descending=True) # shape [num_filtered]
        box = boxes[ordered_inds] # shape [num_filtered, 4]
        conf = conf[ordered_inds] # shape [num_filtered]
        ldmks = ldmks[ordered_inds] # shape [num_filtered, 10]
        keep = nms(box, conf, nms_thresh) # shape [num_kept]
        box = box[keep].cpu().numpy() # shape [num_kept, 4]
        conf = conf[keep].cpu().numpy() # shape [num_kept]
        ldmks = ldmks[keep].cpu().numpy() # shape [num_kept, 10]
        dets = []
        for i in range(box.shape[0]):
            b = box[i]
            score = conf[i].item()
            ldmks[i] = [x for x in ldmks[i]]
            dets.append((int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item()), score, ldmks[i].tolist()))
        return dets

    def remove_prefix(self, state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def check_keys(self, model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> bool:
        ckpt_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        if len(used_pretrained_keys) == 0:
            print('No matching keys found between model and checkpoint.')
            return False
        elif len(missing_keys) > 0:
            print('Missing keys in state_dict:{}'.format(missing_keys))
            return False
        elif len(unused_pretrained_keys) > 0:
            print('Unused checkpoint keys:{}'.format(unused_pretrained_keys))
            return False
        return True
    
    def decode_bbox(self, loc: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
        """Decode locations from predictions using priors to undo the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    
    def decode_ldmk(self, ldmk: torch.Tensor, prior_data: torch.Tensor, var: List[float]) -> torch.Tensor:
        """Decode landmarks from predictions using priors to undo the encoding we did for offset regression at train time.
        Args:
            ldmk (tensor): landmark predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landmark predictions
        """
        return torch.cat((prior_data[:, :2] + ldmk[:, :2] * var[0] * prior_data[:, 2:],
                        prior_data[:, :2] + ldmk[:, 2:4] * var[0] * prior_data[:, 2:],
                        prior_data[:, :2] + ldmk[:, 4:6] * var[0] * prior_data[:, 2:],
                        prior_data[:, :2] + ldmk[:, 6:8] * var[0] * prior_data[:, 2:],
                        prior_data[:, :2] + ldmk[:, 8:10] * var[0] * prior_data[:, 2:],
                        ), dim=1)

    def prior_box(self, image_size: Tuple[int, int], cfg: Dict, device: torch.device) -> torch.Tensor:
        feature_maps = [[ceil(image_size[0]/step), ceil(image_size[1]/step)] for step in cfg['steps']]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = cfg['min_sizes'][k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [x * cfg['steps'][k] / image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * cfg['steps'][k] / image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).to(device).view(-1, 4)
        if cfg['clip']:
            output.clamp_(max=1, min=0)
        return output
