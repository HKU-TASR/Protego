import os
import shutil
from typing import List, Any, Union, Optional, Dict
from collections import OrderedDict
import zipfile

import torch
import torch.nn.functional as F
from torchvision import transforms
import gdown

from FR_DB.ir50_opom.IR import IR_50
from FR_DB.arcface.arcface import Arcface
from FR_DB.adaface import net
from FR_DB.facenet.inception_resnet_v1 import InceptionResnetV1
from FR_DB.magface import iresnet
from FR_DB.magface.iresnet import IResNet
from FR_DB.vit.vit_face import ViT_face
from FR_DB.vit.vits_face import ViTs_face
from . import BASE_PATH

FR_DB_PATH = os.path.join(BASE_PATH, 'FR_DB')
IR50_OPOM_HOME = os.path.join(FR_DB_PATH, 'ir50_opom')
ADAFACE_HOME = os.path.join(FR_DB_PATH, 'adaface')
ARCFACE_HOME = os.path.join(FR_DB_PATH, 'arcface')
FACENET_HOME = os.path.join(FR_DB_PATH, 'facenet')
MAGFACE_HOME = os.path.join(FR_DB_PATH, 'magface')
VIT_HOME = os.path.join(FR_DB_PATH, 'vit')

# Ten training-surrogate models (Table 2). The default intruder is 'ir50_adaface_casia'.
BASIC_POOL = ['ir50_softmax_casia', 'ir50_cosface_casia', 
              'ir50_arcface_casia', 'mobilenet_arcface_casia', 'mobilefacenet_arcface_casia', 
              'ir18_adaface_webface', 'ir50_adaface_ms1mv2', 'ir50_adaface_casia', 'ir50_adaface_webface', 'ir101_adaface_webface']
# Six held-out black-box test models with fundamentally different algorithms/architectures.
SPECIAL_POOL = ['inception_facenet_vgg', 'inception_facenet_casia',
              'ir50_magface_ms1mv2', 'ir100_magface_ms1mv2',
              'vit_cosface_ms1mv2', 'vits_cosface_ms1mv2']
VIT_FAMILY = SPECIAL_POOL[4:6]

def download(path: str, url: str) -> None:
    """
    A crude download util function, only tailoreed for the scenarios this project meets. 

    Args:
        path (str): The local desired path of the weight file
        url (str): The URL of the file to download.
    """
    folder = "/".join(path.split("/")[:-1])+'/'
    if not os.path.exists(path):
        print(f"Weight {path} does not exist. Downloading it from {url}")
        if "uc?id" in url:
            file_id = url.split("uc?id=")[-1]
            downloaded_f = gdown.download(id=file_id, output=folder, quiet=False)
        elif "usp=sharing" in url or "usp=share_link" in url:
            if "file" in url:
                downloaded_f = gdown.download(url=url, output=folder, fuzzy=True, quiet=False)
                if downloaded_f.endswith(".zip"):
                    with zipfile.ZipFile(downloaded_f, 'r') as zip_ref:
                        zip_ref.extractall(folder)
                    os.remove(downloaded_f)
                elif downloaded_f != path:
                    os.rename(downloaded_f, path)
            elif "folder" in url:
                download_fs = gdown.download_folder(url=url, output=folder, quiet=False)
                for f in download_fs:
                    shutil.move(f, os.path.join(folder, f.split("/")[-1]))
                os.removedirs(os.path.join(folder, download_fs[0].split("/")[-2]))
            else:
                raise ValueError(f"Unrecognized URL format: {url}")
    return
    
class FR(object):
    """
    Attributes:
        device (torch.device): The device to run the model on.
        model_name (str): The name of the FR model.
        fr_model (object): The actual FR model object.
        embedding_dim (int): The dimension of the feature vector.
        dis_func (str): The distance function used during training. 
        img_size (int): The expected size of the input image.
        drange (int): The range of pixel values in the expected input image.
        preprocess_method (callable): The preprocessing method for the input image.
        l2norm (bool): Whether the direct output of the model is L2-normalized.

    Methods:
        __init__(model_name: str, device: torch.device):
            Initializes the FR model with the specified model name and device.
        __call__(imgs: torch.Tensor) -> torch.Tensor:
            Combines the preprocessing and feature extraction steps.
        extract_features(images: torch.Tensor) -> torch.Tensor:
            Extracts features from a batch of images.
        preprocess(images: torch.Tensor) -> torch.Tensor:
            Preprocesses a batch of images for the FR model.
    """
    def __init__(self, model_name: str = 'inception_facenet_vgg', device: torch.device = torch.device('cpu')):
        """
        Init the FR model with the specified model name and device.

        Args:
            model_name (str): The name of the FR model. Default is 'inception_facenet_vgg'. Options are:
            (- 'model_name': Neural Structure + Loss + Dataset)
            - 'inception_facenet_vgg': InceptionResnetV1 + Facenet + VGGFace2
            - 'inception_facenet_casia': InceptionResnetV1 + Facenet + CASIA-WebFace
            - 'ir50_softmax_casia': Improved ResNet50 + Softmax + CASIA-WebFace
            - 'ir50_cosface_casia': Improved ResNet50 + CosFace + CASIA-WebFace
            - 'ir50_arcface_casia': Improved ResNet50 + ArcFace + CASIA-WebFace
            - 'ir50_adaface_casia': Improved ResNet50 + AdaFace + CASIA-WebFace
            - 'mobilenet_arcface_casia': MobileNet + ArcFace + CASIA-WebFace
            - 'mobilefacenet_arcface_casia': MobileFacenet + ArcFace + CASIA-WebFace
            - 'ir18_adaface_webface': Improved ResNet18 + AdaFace + WebFace4M
            - 'ir50_adaface_webface': Improved ResNet50 + AdaFace + WebFace4M
            - 'ir101_adaface_webface': Improved ResNet100 + AdaFace + WebFace4M
            - 'ir50_adaface_ms1mv2': Improved ResNet50 + AdaFace + MS1MV2
            - 'inception_facenet_vgg': InceptionResnetV1 + Facenet + VGGFace2
            - 'inception_facenet_casia': InceptionResnetV1 + Facenet + CASIA-WebFace
            - 'ir50_magface_ms1mv2': Improved ResNet50 + MagFace + MS1MV2
            - 'ir100_magface_ms1mv2': Improved ResNet100 + MagFace + MS1MV2
            device (torch.device): The device to run the model on. Default is CPU.

        Raises:
            ValueError: If the model name is not valid.

        Example:
            >>> import torch
            >>> from FacialRecognition import * 
            >>> model_name = BASIC_POOL[0] # Choose a model from the model pool
            >>> fr = FR(model_name=model_name, device=torch.device('cpu'))
            >>> img = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8) # Shape: (B, 3, H, H), Range: [0, 255], RGB, dtype: torch.uint8
            >>> img = img.float() # Convert to float32
            >>> features = fr(img) 
            >>> print(features.shape) # Should be (1, 512) for most models. For 'mobilefacenet_arcface_casia', it should be (1, 128)
        """
        self.device = device
        self.model_name = model_name

        if self.model_name == 'ir50_softmax_casia': # Improved ResNet50 + Softmax + CASIA-WebFace
            self.fr_model = IR50_OPOM(loss='softmax', device=self.device)
        elif self.model_name == 'ir50_cosface_casia': # Improved ResNet50 + CosFace + CASIA-WebFace
            self.fr_model = IR50_OPOM(loss='cosface', device=self.device)
        elif self.model_name == 'ir50_arcface_casia': # Improved ResNet50 + ArcFace + CASIA-WebFace
            self.fr_model = ArcFace(backbone='iresnet50', device=self.device)
        elif self.model_name == 'ir50_adaface_casia': # Improved ResNet50 + AdaFace + CASIA-WebFace
            self.fr_model = AdaFace(backbone='ir_50_casia', device=self.device)
        elif self.model_name == 'mobilenet_arcface_casia': # MobileNet + ArcFace + CASIA-WebFace
            self.fr_model = ArcFace(backbone='mobilenetv1', device=self.device)
        elif self.model_name == 'mobilefacenet_arcface_casia': # MobileFacenet + ArcFace + CASIA-WebFace
            self.fr_model = ArcFace(backbone='mobilefacenet', device=self.device)
        elif self.model_name == 'ir18_adaface_webface': # Improved ResNet18 + AdaFace + WebFace4M
            self.fr_model = AdaFace(backbone='ir_18_web', device=self.device)
        elif self.model_name == 'ir50_adaface_webface': # Improved ResNet50 + AdaFace + WebFace4M
            self.fr_model = AdaFace(backbone='ir_50_web', device=self.device)
        elif self.model_name == 'ir101_adaface_webface': # Improved ResNet101 + AdaFace + WebFace4M
            self.fr_model = AdaFace(backbone='ir_101_web', device=self.device)
        elif self.model_name == 'ir50_adaface_ms1mv2': # Improved ResNet50 + AdaFace + MS1MV2
            self.fr_model = AdaFace(backbone='ir_50_ms1mv2', device=self.device)
        elif self.model_name == 'inception_facenet_vgg': # InceptionResnetV1 + Facenet + VGGFace2
            self.fr_model = InceptionFacenet(pretrained='vggface2', device=self.device)
        elif self.model_name == 'inception_facenet_casia': # InceptionResnetV1 + Facenet + CASIA-WebFace
            self.fr_model = InceptionFacenet(pretrained='casia-webface', device=self.device)
        elif self.model_name == 'ir50_magface_ms1mv2': # Improved ResNet50 + MagFace + MS1MV2
            self.fr_model = Magface(arch='iresnet50', device=self.device)
        elif self.model_name == 'ir100_magface_ms1mv2': # Improved ResNet100 + MagFace + MS1MV2
            self.fr_model = Magface(arch='iresnet100', device=self.device)
        elif self.model_name == 'vit_cosface_ms1mv2': # ViT + CosFace + MS1MV2
            self.fr_model = ViT(backbone='vit', device=self.device)
        elif self.model_name == 'vits_cosface_ms1mv2': # ViTs + CosFace + MS1MV2
            self.fr_model = ViT(backbone='vits', device=self.device)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")
        
        self.embedding_dim = self.fr_model.embedding_dim
        self.dis_func = self.fr_model.dis_func
        self.img_size = self.fr_model.img_size
        self.drange = self.fr_model.drange
        self.preprocess_method = self.fr_model.preprocessing
        self.l2norm = self.fr_model.l2norm

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Combines the preprocessing and feature extraction steps.

        Args:
            images (torch.Tensor): A batch of images with shape (B, 3, H, H), dtype: float32, RGB format.

        Returns:
            torch.Tensor: Features with shape (B, C). Refer to the `extract_features` method for details.
        """
        return self.extract_features(self.preprocess(imgs))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from a batch of images.

        Args:
            images (torch.Tensor): A batch of preprocessed images with shape (B, 3, H, W).

        Returns:
            torch.Tensor: Features with shape (B, C). 
                - 'ir50_softmax_casia' : (B, 512)
                - 'ir50_cosface_casia' : (B, 512)
                - 'ir50_arcface_casia' : (B, 512)
                - 'ir50_adaface_casia' : (B, 512)
                - 'mobilenet_arcface_casia' : (B, 512)
                - 'mobilefacenet_arcface_casia' : (B, 128)
                - 'ir18_adaface_webface' : (B, 512)
                - 'ir50_adaface_webface' : (B, 512)
                - 'ir101_adaface_webface' : (B, 512)
                - 'ir50_adaface_ms1mv2' : (B, 512)
                - 'inception_facenet_vgg' : (B, 512)
                - 'inception_facenet_casia' : (B, 512)
        """
        return self.fr_model.forward(images)
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses a batch of images for the FR model.

        Args:
            images (torch.Tensor): A batch of images with shape (B, 3, H, H), dtype: float32, RGB format.

        Returns:
            torch.Tensor: Preprocessed images with shape (B, 3, H', H'). Range [0, 1] or [-1, 1] or [0, 255], dtype: float32., RGB or BGR format.
        """
        if images.max() >= 2 and self.drange == 1:
            rescaled_images = images.float() / 255.
        elif images.max() >= 2 and self.drange == 255:
            rescaled_images = images.float()
        elif images.max() < 2 and self.drange == 1:
            rescaled_images = images.float()
        elif images.max() < 2 and self.drange == 255:
            rescaled_images = images.float() * 255.
        preprocessed = torch.stack([self.preprocess_method(img) for img in rescaled_images])
        return preprocessed
    
class Magface(object):
    """
    Model: IResNet50/IResNet100
    Datasets: MS1MV2
    Loss Function: MagFace
    Source: https://github.com/IrvingMeng/MagFace

    Args:
        arch (str): Backbone model name. Options are 'iresnet50', 'iresnet100'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, arch: str, device: torch.device):
        self.embedding_dim = 512
        self.path_dict = {
            'iresnet100': os.path.join(MAGFACE_HOME, 'pretrained/magface_iresnet100_MS1MV2_ddp.pth'),
            'iresnet50': os.path.join(MAGFACE_HOME, 'pretrained/magface_iresnet50_MS1MV2_ddp_fp32.pth')
        }
        self.url_dict = {
            'iresnet100': "https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view?usp=sharing",
            'iresnet50': "https://drive.google.com/file/d/1QPNOviu_A8YDk9Rxe8hgMIXvDKzh6JMG/view?usp=sharing"
        }
        download(self.path_dict[arch], self.url_dict[arch])
        self.model = MagfaceNetworkBuilder(arch=arch, embedding_size=self.embedding_dim, weight_path=self.path_dict[arch], device=device)
        self.model = self.model.to(device)
        self.model = self.model.eval() 

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
            transforms.Lambda(lambda x: x[[2, 1, 0], :, :]) # ! Convert RGB to BGR for MagFace
        ])
        self.img_size = 112 
        self.drange = 1 
        self.dis_func = "cosine" 
        self.l2norm = False

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(imgs)
    
class MagfaceNetworkBuilder(torch.nn.Module):
    def __init__(self, arch:str, embedding_size:int, weight_path: str, device: torch.device):
        super(MagfaceNetworkBuilder, self).__init__()
        self.model = self.get_net(arch, embedding_size)
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        self.load_weights(weight_path, device)

    def get_net(self, arch:str, embedding_size:int) -> IResNet:
        if arch == 'iresnet34':
            return iresnet.iresnet34(pretrained=False, num_classes=embedding_size,)
        elif arch == 'iresnet18':
            return iresnet.iresnet18(pretrained=False, num_classes=embedding_size,)
        elif arch == 'iresnet50':
            return iresnet.iresnet50(pretrained=False, num_classes=embedding_size,)
        elif arch == 'iresnet100':
            return iresnet.iresnet100(pretrained=False, num_classes=embedding_size,)
        else:
            raise ValueError(f"Unknown architecture for Magface: {arch}")
        
    def load_weights(self, weight_path: str, device: torch.device) -> None:
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        ckpts = torch.load(weight_path, map_location=device, weights_only=False) # ! Note that the older torch.load function set `weights_only=False` by default, so we need to set it explicitly
        #print(ckpts['state_dict'].keys())
        #print(self.model.state_dict().keys())
        state_dict = self.clean_dict_inf(ckpts['state_dict'])
        model_dict = self.model.state_dict()
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)
    
    def clean_dict_inf(self, state_dict: OrderedDict) -> OrderedDict:
        _state_dict = OrderedDict()
        for k, v in state_dict.items():
            # # assert k[0:1] == 'features.module.'
            #new_k = 'features.'+'.'.join(k.split('.')[2:])
            new_k = '.'.join(k.split('.')[2:]) # ! Different from the original mapping due to different load method
            if new_k in self.model.state_dict().keys() and v.size() == self.model.state_dict()[new_k].size():
                _state_dict[new_k] = v
            # assert k[0:1] == 'module.features.'
            new_kk = '.'.join(k.split('.')[1:])
            if new_kk in self.model.state_dict().keys() and v.size() == self.model.state_dict()[new_kk].size():
                _state_dict[new_kk] = v
        num_model = len(self.model.state_dict().keys())
        num_ckpt = len(_state_dict.keys())
        if num_model != num_ckpt:
            raise ValueError(f"Model state dict size {num_model} does not match checkpoint state dict size {num_ckpt}.")
        return _state_dict
        
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model(imgs)

class InceptionFacenet(object):
    """
    Model: InceptionResnetV1
    Datasets: VGGFace2/CASIA-WebFace
    Loss Function: Facenet
    Source: https://github.com/timesler/facenet-pytorch

    Args:
        pretrained (str): Pretrained model name. Options are 'casia-webface', 'vggface2'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, pretrained: str, device: torch.device):
        self.model = InceptionResnetV1(classify=False, pretrained=pretrained).to(device)
        self.model = self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Lambda(lambda x: x * 255.0),
            transforms.Lambda(lambda x: (x - 127.5) / 128.) # ! Same as the original normalization in facenet-pytorch
        ])
        self.img_size = 160
        self.drange = 1
        self.dis_func = "euclidean"
        self.embedding_dim = 512
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model(imgs)

class IR50_OPOM(object):
    """
    Model: Improved ResNet50
    Datasets: CASIA-WebFace
    Loss Function: Softmax/CosFace
    Source: https://github.com/zhongyy/OPOM

    Args:
        weight_path (str): Path to the model weights.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, loss: str, device: torch.device):
        self.path_dict = {
            'softmax': os.path.join(IR50_OPOM_HOME, 'pretrained/Backbone_IR_50_Epoch_94_Batch_180000_Time_2020-03-30-01-55_checkpoint.pth'),
            'cosface': os.path.join(IR50_OPOM_HOME, 'pretrained/Backbone_IR_50_Epoch_74_Batch_140000_Time_2020-05-27-21-38_checkpoint.pth')
        }
        self.url_dict = {
            'softmax': "https://drive.google.com/file/d/10YKGalAc3B0ywudFMUp61HAMij8p9xhW/view?usp=sharing",
            'cosface': "https://drive.google.com/file/d/1NUm9jYoC-q0f2Yz8rYky8NdgwRgu4-vr/view?usp=sharing"
        }
        download(self.path_dict[loss], self.url_dict[loss])
        weight_path = self.path_dict[loss]
        self.model = IR_50([112, 112])
        self.model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
        self.model = self.model.to(device)
        self.model = self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112))
            #transforms.Lambda(lambda x: x * 255.0)
        ])
        self.img_size = 112
        self.drange = 255
        self.dis_func = "cosine" if loss == 'cosface' else 'euclidean'
        self.embedding_dim = 512 
        self.l2norm = False 

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(imgs)
    
class ArcFace(object):
    """
    Model: IR50/MobileNetV1/MobileFacenet
    Datasets: CASIA-WebFace
    Loss Function: ArcFace
    Source: https://github.com/bubbliiiing/arcface-pytorch

    Args:
        backbone (str): Backbone model name. Options are 'iresnet50', 'mobilenetv1', 'mobilefacenet'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, backbone: str, device: torch.device):
        self.path_dict = {
            'iresnet50': os.path.join(ARCFACE_HOME, 'pretrained/arcface_iresnet50.pth'),
            'mobilenetv1': os.path.join(ARCFACE_HOME, 'pretrained/arcface_mobilenet_v1.pth'),
            'mobilefacenet': os.path.join(ARCFACE_HOME, 'pretrained/arcface_mobilefacenet.pth')
        }
        self.url_dict = {
            'iresnet50': "https://drive.google.com/file/d/1V-dNjoNaXrEVlt5G9DIP5kmIjo5s9Pbk/view?usp=sharing",
            'mobilenetv1': "https://drive.google.com/file/d/13qtdGZc7YmovAnk1Sf0K5AEG8fiCu_nC/view?usp=sharing", 
            'mobilefacenet': "https://drive.google.com/file/d/1YZN0K-eEB3IF-HDZdXfXR-ztxTMPnYq2/view?usp=sharing"
        }
        download(self.path_dict[backbone], self.url_dict[backbone])
        model_path = self.path_dict[backbone]

        self.model = Arcface(backbone=backbone, mode='predict')
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
        self.model = self.model.to(device)
        self.model = self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112), interpolation=transforms.InterpolationMode.BICUBIC), # ! Same as the original ArcFace preprocessing
            transforms.Lambda(lambda x: (x - 0.5) / 0.5)
        ])
        self.img_size = 112
        self.drange = 1
        self.dis_func = "cosine"
        self.embedding_dim = 512 if not backbone == 'mobilefacenet' else 128
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(imgs)
    
class AdaFace(object):
    """
    Model: Improved ResNet18/50/101
    Datasets: WebFace4M/CASIA-WebFace/MS1MV2
    Loss Function: AdaFace
    Source: https://github.com/mk-minchul/AdaFace

    Args:
        backbone (str): Backbone model name. Options are 'ir_18_web', 'ir_50_web', 'ir_50_casia', 'ir_101_web', 'ir_50_ms1mv2'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, backbone: str, device: torch.device):
        self.path_dict = {
            'ir_18_web': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir18_webface4m.ckpt'),
            'ir_50_casia': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_casia.ckpt'),
            'ir_50_web': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_webface4m.ckpt'),
            'ir_101_web': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir101_webface4m.ckpt'),
            'ir_50_ms1mv2': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_ms1mv2.ckpt'),
        }
        self.url_dict = {
            'ir_18_web': "https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view?usp=sharing",
            'ir_50_casia': "https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing",
            'ir_50_web': "https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing",
            'ir_101_web': "https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing",
            'ir_50_ms1mv2': "https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing"
        }
        if backbone in self.url_dict.keys():
            download(self.path_dict[backbone], self.url_dict[backbone])
        model_path = self.path_dict[backbone]
        self.model = net.build_model("_".join(backbone.split('_')[:2]))
        statedict = torch.load(model_path, weights_only=False, map_location=device)['state_dict']
        model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model = self.model.to(device)
        self.model = self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Lambda(lambda x: (x - 0.5) / 0.5), 
            transforms.Lambda(lambda x: x[[2, 1, 0], :, :])
        ])
        self.img_size = 112
        self.drange = 1
        self.dis_func = "cosine"
        self.embedding_dim = 512 
        self.l2norm = True 

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model(imgs)[0]

class ViT(object):
    """
    Model: ViT/ViTs (Pure and simplest ViT. ViT does hard patching while ViTs does overlapping/soft patching)
    Datasets: MS1MV2
    Loss Function: CosFace
    Source: https://github.com/zhongyy/Face-Transformer (arXiv)

    Args:
        backbone (str): Backbone model name. Options are 'vit', 'vits'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, backbone: str, device: torch.device):
        self.path_dict = {
            'vit': os.path.join(VIT_HOME, 'pretrained/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth'),
            'vits': os.path.join(VIT_HOME, 'pretrained/Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth')
        }
        self.url_dict = {
            'vit': "https://drive.google.com/file/d/1OZRU430CjABSJtXU0oHZHlxgzXn6Gaqu/view?usp=share_link", 
            'vits': "https://drive.google.com/file/d/1U7c_ojiuRPBfolvziB_VthksABHaFKud/view?usp=share_link"
        }
        download(self.path_dict[backbone], self.url_dict[backbone])
        model_path = self.path_dict[backbone]
        if backbone == 'vit':
            self.model = ViT_face(
                image_size=112, 
                patch_size=8, 
                loss_type='CosFace', 
                GPU_ID=device, 
                num_class=93431, 
                dim=512, 
                depth=20, 
                heads=8, 
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1)
        elif backbone == 'vits':
            self.model = ViTs_face(
                loss_type='CosFace',
                GPU_ID=device,
                num_class=93431,
                image_size=112,
                patch_size=8,
                ac_patch_size=12,
                pad=4,
                dim=512,
                depth=20,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model = self.model.to(device).eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112))
        ])
        self.img_size = 112
        self.drange = 255
        self.dis_func = "cosine"
        self.embedding_dim = 512
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        flipped_imgs = torch.flip(imgs, dims=[3])
        return F.normalize(self.model(imgs) + self.model(flipped_imgs))
