from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        """
        Custom flatten layer that reshapes tensor from [B, C, H, W] to [B, C*H*W].
        This implementation is specifically designed for compatibility with the pretrained MTCNN weights.
        """
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: tensor of shape [B, C, H, W]
               
        Returns:
            torch.Tensor: Flattened tensor of shape [B, C*H*W]
        """
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class PNet(nn.Module):
    """
    Proposal Network (PNet)
    
    PNet is a fully convolutional network that rapidly scans the input image
    to generate candidate facial bounding boxes and their confidence scores.
    It operates on multiple scales to detect faces of different sizes.
    
    Architecture:
    - 3 convolutional layers with PReLU activation
    - 1 max pooling layer for downsampling
    - 2 output branches: face classification (2 classes) and bounding box regression (4 coords)
    
    Input size calculation:
    For input size HxW:
    - After conv1 (3x3): H-2, W-2
    - After pool1 (2x2): ceil((H-2)/2), ceil((W-2)/2)  
    - After conv2 (3x3): ceil((H-2)/2)-2, ceil((W-2)/2)-2
    - After conv3 (3x3): ceil((H-2)/2)-4, ceil((W-2)/2)-4
    """
    
    def __init__(self, weight_path: str):
        """
        Initialize PNet with pretrained weights.
        
        Args:
            weight_path: Path to the .npy file containing pretrained weights
        """
        super(PNet, self).__init__()

        # Feature extraction backbone
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),    # 3x3 conv, 3->10 channels
            ('prelu1', nn.PReLU(10)),              # Parametric ReLU activation
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),  # 2x2 max pooling

            ('conv2', nn.Conv2d(10, 16, 3, 1)),   # 3x3 conv, 10->16 channels  
            ('prelu2', nn.PReLU(16)),              # Parametric ReLU activation

            ('conv3', nn.Conv2d(16, 32, 3, 1)),   # 3x3 conv, 16->32 channels
            ('prelu3', nn.PReLU(32))               # Parametric ReLU activation
        ]))

        # Output heads
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)   # Face classification: background/face
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)   # Bounding box regression: x, y, w, h

        # Load pretrained weights
        weights = np.load(weight_path, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """  
        Args:
            x: Input image tensor of shape [B, 3, H, W]
        
        Returns:
            Tuple containing:
            - b: Bounding box regression tensor of shape [B, 4, H', W']
                 where each location contains [dx, dy, dw, dh] offsets
            - a: Face classification probabilities of shape [B, 2, H', W']
                 where channel 0 = background prob, channel 1 = face prob
        """
        # Extract features through backbone
        x = self.features(x)
        
        # Generate outputs
        a = self.conv4_1(x)  # Face classification logits
        b = self.conv4_2(x)  # Bounding box regression
        
        # Apply softmax to get face probabilities
        a = F.softmax(a, dim=1)
        
        return b, a


class RNet(nn.Module):
    """
    Refine Network (RNet) - Second stage of MTCNN cascade.
    
    RNet takes candidate bounding boxes from PNet and further refines them
    by rejecting false positives and improving localization accuracy.
    It operates on 24x24 pixel image patches extracted from candidate regions.
    
    Architecture:
    - 3 convolutional layers with PReLU activation and max pooling
    - 1 fully connected layer for feature extraction
    - 2 output branches: face classification and bounding box regression
    
    The network processes fixed-size 24x24 input patches, making it more
    discriminative than PNet for face/non-face classification.
    """

    def __init__(self, weight_path: str):
        """
        Initialize RNet with pretrained weights.
        
        Args:
            weight_path: Path to the .npy file containing pretrained weights
        """
        super(RNet, self).__init__()

        # Feature extraction backbone
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),    # 3x3 conv, 3->28 channels
            ('prelu1', nn.PReLU(28)),              # Parametric ReLU activation
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),  # 3x3 max pooling, stride 2

            ('conv2', nn.Conv2d(28, 48, 3, 1)),   # 3x3 conv, 28->48 channels
            ('prelu2', nn.PReLU(48)),              # Parametric ReLU activation  
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),  # 3x3 max pooling, stride 2

            ('conv3', nn.Conv2d(48, 64, 2, 1)),   # 2x2 conv, 48->64 channels
            ('prelu3', nn.PReLU(64)),              # Parametric ReLU activation

            ('flatten', Flatten()),                # Custom flatten layer
            ('conv4', nn.Linear(576, 128)),        # Fully connected layer, 576->128
            ('prelu4', nn.PReLU(128))              # Parametric ReLU activation
        ]))

        # Output heads  
        self.conv5_1 = nn.Linear(128, 2)    # Face classification: background/face
        self.conv5_2 = nn.Linear(128, 4)    # Bounding box regression: x, y, w, h

        # Load pretrained weights
        weights = np.load(weight_path, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RNet.
        
        Args:
            x: Input image patches tensor of shape [B, 3, 24, 24]
               Each patch is a candidate face region from PNet
        
        Returns:
            Tuple containing:
            - b: Bounding box regression tensor of shape [B, 4]
                 containing [dx, dy, dw, dh] refinement offsets for each patch
            - a: Face classification probabilities of shape [B, 2]
                 where a[:, 0] = background prob, a[:, 1] = face prob
        """
        # Extract features through backbone
        x = self.features(x)
        
        # Generate outputs
        a = self.conv5_1(x)  # Face classification logits
        b = self.conv5_2(x)  # Bounding box regression
        
        # Apply softmax to get face probabilities
        a = F.softmax(a, dim=1)
        
        return b, a


class ONet(nn.Module):
    """
    Output Network (ONet) - Final stage of MTCNN cascade.
    
    ONet performs the most precise face detection and additionally predicts
    facial landmarks (eye centers, nose tip, mouth corners). It operates on
    48x48 pixel image patches from refined candidate regions.
    
    Architecture:
    - 4 convolutional layers with PReLU activation and max pooling
    - 1 fully connected layer with dropout for regularization  
    - 3 output branches: face classification, bounding box regression, and landmarks
    
    The network provides the final face detection results along with 5 facial
    landmarks for face alignment applications.
    """

    def __init__(self, weight_path: str):
        """
        Initialize ONet with pretrained weights.
        
        Args:
            weight_path: Path to the .npy file containing pretrained weights
        """
        super(ONet, self).__init__()

        # Feature extraction backbone
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),    # 3x3 conv, 3->32 channels
            ('prelu1', nn.PReLU(32)),              # Parametric ReLU activation
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),  # 3x3 max pooling, stride 2

            ('conv2', nn.Conv2d(32, 64, 3, 1)),   # 3x3 conv, 32->64 channels
            ('prelu2', nn.PReLU(64)),              # Parametric ReLU activation
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),  # 3x3 max pooling, stride 2

            ('conv3', nn.Conv2d(64, 64, 3, 1)),   # 3x3 conv, 64->64 channels
            ('prelu3', nn.PReLU(64)),              # Parametric ReLU activation
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),  # 2x2 max pooling, stride 2

            ('conv4', nn.Conv2d(64, 128, 2, 1)),  # 2x2 conv, 64->128 channels
            ('prelu4', nn.PReLU(128)),             # Parametric ReLU activation

            ('flatten', Flatten()),                # Custom flatten layer
            ('conv5', nn.Linear(1152, 256)),       # Fully connected layer, 1152->256
            ('drop5', nn.Dropout(0.25)),           # Dropout for regularization
            ('prelu5', nn.PReLU(256)),             # Parametric ReLU activation
        ]))

        # Output heads
        self.conv6_1 = nn.Linear(256, 2)     # Face classification: background/face
        self.conv6_2 = nn.Linear(256, 4)     # Bounding box regression: x, y, w, h  
        self.conv6_3 = nn.Linear(256, 10)    # Facial landmarks: 5 points * (x, y) = 10 coords

        # Load pretrained weights
        weights = np.load(weight_path, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ONet.
        
        Args:
            x: Input image patches tensor of shape [B, 3, 48, 48]
               Each patch is a refined candidate face region from RNet
        
        Returns:
            Tuple containing:
            - c: Facial landmarks tensor of shape [B, 10]
                 containing [x1, y1, x2, y2, ..., x5, y5] coordinates for:
                 left eye, right eye, nose, left mouth corner, right mouth corner
            - b: Bounding box regression tensor of shape [B, 4]
                 containing [dx, dy, dw, dh] final refinement offsets
            - a: Face classification probabilities of shape [B, 2]
                 where a[:, 0] = background prob, a[:, 1] = face prob
        """
        # Extract features through backbone
        x = self.features(x)
        
        # Generate outputs
        a = self.conv6_1(x)  # Face classification logits
        b = self.conv6_2(x)  # Bounding box regression
        c = self.conv6_3(x)  # Facial landmarks regression
        
        # Apply softmax to get face probabilities
        a = F.softmax(a, dim=1)
        
        return c, b, a
