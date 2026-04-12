from typing import Dict, Any, Tuple, Union
import io
import datetime
from fractions import Fraction
import random

import torch
import torch.nn.functional as F
import av
import cv2

from DiffJPEG.DiffJPEG import DiffJPEG

def gaussian_filter(imgs:torch.Tensor, kernel_size:int=5, sigma:float=1.0, differentiable: bool = False) -> torch.Tensor:
    """
    A differentiable Gaussian filter

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        sigma (float): Standard deviation for Gaussian distribution
        differentiable (bool): Placeholder for differentiable flag, not used in this implementation

    Returns:
        Tensor: Filtered image tensor of shape [B, C, H, W]
    """
    device = imgs.device
    # Create a 2D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    x = x.to(device).float()  # Ensure x is on the same device as imgs
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()  # Normalize
    kernel_1d = gauss.unsqueeze(0)  # [1, kernel_size]
    kernel_2d = torch.mm(kernel_1d.T, kernel_1d)  # [kernel_size, kernel_size]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size, kernel_size]

    # Use the kernel for convolution
    channels = imgs.shape[1]
    kernel = kernel_2d.repeat(channels, 1, 1, 1)  # [C, 1, kernel_size, kernel_size]
    padding = kernel_size // 2
    filtered_image = F.conv2d(imgs, kernel, padding=padding, groups=channels)
    return filtered_image

def median_filter(imgs: torch.Tensor, kernel_size: int = 3, differentiable: bool = False) -> torch.Tensor:
    """
    A median filter with differentiable and non-differentiable modes.

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        kernel_size (int): Size of the median filter kernel (must be odd)
        differentiable (bool): If True, use a differentiable approximation of the median filter

    Returns:
        Tensor: Filtered image tensor of shape [B, C, H, W]
    """
    device = imgs.device
    padding = kernel_size // 2
    imgs_padded = F.pad(imgs, (padding, padding, padding, padding), mode='reflect')  # [B, C, H+2*padding, W+2*padding]

    B, C, H, W = imgs.shape
    unfolded = F.unfold(imgs_padded, kernel_size, stride=1)  # [B, C*kernel_size*kernel_size, H*W]
    unfolded = unfolded.view(B, C, kernel_size * kernel_size, H, W)  # [B, C, kernel_size*kernel_size, H, W]

    if differentiable:
        # Use softmax to approximate the median in a differentiable way
        weights = torch.softmax(unfolded, dim=2)  # [B, C, kernel_size*kernel_size, H, W]
        median_approx = (unfolded * weights).sum(dim=2)  # [B, C, H, W]
        return median_approx
    else:
        # Use the median function to compute the median across the kernel dimension
        median = unfolded.median(dim=2).values  # [B, C, H, W]
        return median

def jpeg(imgs:torch.Tensor, quality:int=80, differentiable:bool=False) -> torch.Tensor:
    """
    Differentiable JPEG compression adopted from https://github.com/mlomnitz/DiffJPEG. 

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]. Note that if H or W is not divisible by 16, the image will be padded to the next multiple of 16
        quality (int): Quality factor for JPEG compression
        differentiable (bool): If True, use differentiable rounding

    Returns:
        Tensor: Compressed image tensor of shape [B, C, H, W]
        Note: Despite the padding, the output tensor will have the same shape as the input tensor by direct cutting. 
    """
    device = imgs.device
    B, C, H, W = imgs.shape
    # Pad to multiples of 16 so that after 2x chroma subsampling (Cb/Cr at H/2, W/2),
    # dimensions are still divisible by 8 for 8x8 block splitting.
    block = 16
    pad_h = (block - (H % block)) % block
    pad_w = (block - (W % block)) % block

    if pad_h != 0 or pad_w != 0:
        # Pad to multiples of 8 for DiffJPEG block operations
        # F.pad expects pads in (left, right, top, bottom)
        imgs_padded = F.pad(imgs, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = H + pad_h, W + pad_w
        compressor = DiffJPEG(height=H_pad, width=W_pad, differentiable=differentiable, quality=quality).to(device)
        out = compressor.forward(imgs_padded)
        # Crop back to original size
        return out[:, :, :H, :W]
    else:
        compressor = DiffJPEG(height=H, width=W, differentiable=differentiable, quality=quality).to(device)
        return compressor.forward(imgs)

def resize(imgs: torch.Tensor, resz_percentage: float, mode: str, differentiable: bool = False) -> torch.Tensor:
    """
    Resize and restore images

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        resz_percentage (float): Percentage to resize the image (e.g., 0.5 for 50%)
        mode (str): Interpolation mode ('bicubic', 'bilinear', etc.)
        differentiable (bool): Placeholder for differentiable flag, not used in this implementation
        
    Returns:
        Tensor: Resized and restored image tensor of shape [B, C, H, W]
    """
    B, C, H, W = imgs.shape
    new_h = int(H * resz_percentage)
    new_w = int(W * resz_percentage)
    resized_imgs = F.interpolate(imgs, size=(new_h, new_w), mode=mode, align_corners=False)
    restored_imgs = F.interpolate(resized_imgs, size=(H, W), mode=mode, align_corners=False)
    return restored_imgs

def quantize(imgs: torch.Tensor, precision: str, diff_method: str = 'ste', differentiable: bool = False) -> torch.Tensor:
    """
    Quantize images using specified method. Assuming that purpose is to reduce precision.

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1], dtype: float32
        precision (str): Precision level ('uint8', 'float16')
        diff_method (str): Quantization method ('ste', 'noise', 'soft_{K}') for differentiable quantization
        differentiable (bool): If True, use differentiable quantization

    Returns:
        Tensor: Quantized image tensor of shape [B, C, H, W].
         - uint8, [0, 255] if not differentiable and precision is 'uint8'
         - float32, [0 ,255] if differentiable and precision is 'uint8'
         - float16, [0, 1] if precision is 'float16'

    Raises:
        ValueError: If the specified method is not supported
    """
    device = imgs.device
    if precision == 'float16':
        return imgs.to(torch.float16).float()
    elif not differentiable and precision == 'uint8':
        return imgs.mul(255).to(torch.uint8).float()
    elif differentiable and precision == 'uint8':
        imgs_scaled = imgs.mul(255.)
        """if diff_method in ['noise']:
            rand_generator = torch.Generator(device=device)
            rand_generator.manual_seed(42)"""
        if diff_method == 'ste':
            quantized = (imgs_scaled.round() - imgs_scaled).detach() + imgs_scaled
        elif diff_method == 'noise':
            noise = torch.empty_like(imgs_scaled).to(device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
            quantized = imgs_scaled + noise
        elif 'soft' in diff_method:
            k = int(diff_method.split('_')[-1])
            floor = torch.floor(imgs_scaled)
            quantized = floor + torch.sigmoid(k * (imgs_scaled - floor - 0.5))
        elif diff_method == 'sine':
            quantized = imgs_scaled - torch.sin(2 * torch.pi * imgs_scaled) / (2 * torch.pi)
        else:
            raise ValueError(f"Unsupported quantization method: {diff_method}")
        """# Debug: Save some quantized images
        for idx, q in enumerate(quantized):
            _quantized = cv2.cvtColor(torch.clamp(q, 0, 255).detach().permute(1, 2, 0).cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR)
            _quantized = cv2.putText(_quantized, f"Quantized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _orig = cv2.cvtColor(torch.clamp(imgs_scaled[idx], 0, 255).detach().permute(1, 2, 0).cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR)
            _orig = cv2.putText(_orig, f"Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _frame = cv2.hconcat([_orig, _quantized])
            cv2.imwrite(f"/home/zlwang/ProtegoPlus/trash/quantize/quantized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png", _frame)"""
        return torch.clamp(quantized, 0, 255)
    else:
        raise ValueError(f"Unsupported precision and differentiable combination: {precision}, {differentiable}")

def vid_compress(imgs: torch.Tensor, codec: str, crf: int = 32, preset: str = 'faster', differentiable: bool = False) -> torch.Tensor:
    """
    Video compression with H.265/HEVC using PyAV. 

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        codec (str): Codec to use ('h264', 'h265', 'hevc', 'avc') ('hevc' and 'h265' are equivalent, 'avc' and 'h264' are equivalent)
        crf (int): Constant Rate Factor for controlling quality (lower means better quality)
        preset (str): Preset for encoding speed vs. compression ratio ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')
            - 'ultrafast': Lowest compression, fastest encoding
            - 'medium': Default preset
            - 'veryslow': Highest compression, slowest encoding
        differentiable (bool): Placeholder for differentiable flag, not used in this implementation
    
    Returns:
        Tensor: Compressed image tensor of shape [B, C, H, W]

    Raises:
        NotImplementedError: Differentiable H.264 compression is not implemented
    """
    if differentiable:
        print("Warning: Differentiable H.264 compression is not implemented. Returning original images.")
        return imgs
    else:
        B, _, H, W = imgs.shape
        mem = io.BytesIO()
        fmt = 'h264' if codec in ['h264', 'avc'] else 'hevc'
        enc_name = 'libx264' if fmt == 'h264' else 'libx265'
        fps = 30
        out = av.open(mem, mode='w', format=fmt)
        stream = out.add_stream(enc_name, rate=fps, options={'crf': str(crf), 'preset': preset})
        stream.width = W
        stream.height = H
        stream.pix_fmt = 'yuv420p'
        # Ensure the stream has a valid time base; some FFmpeg/pyAV combos leave it as None until later
        #if getattr(stream, 'time_base', None) is None:
        stream.time_base = Fraction(1, fps)
        for i in range(B):
            frame_rgb = (imgs[i].detach().mul(255.0)).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            vf = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            vf = vf.reformat(width=W, height=H, format='yuv420p')
            vf.pts = i
            # Only set frame time_base if the stream has one; otherwise skip to avoid None assignment errors
            #if getattr(stream, 'time_base', None) is not None:
            vf.time_base = stream.time_base
            for packet in stream.encode(vf):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
        out.close()
        bitstream = mem.getvalue()
        frames_out = []
        inp = av.open(io.BytesIO(bitstream), mode='r', format=fmt)
        vstream = inp.streams.video[0]
        for frame in inp.decode(vstream):
            rgb = frame.to_ndarray(format='rgb24')
            t = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0  # [3,H,W] float[0,1] on CPU
            frames_out.append(t)
        inp.close()
        """for idx, frame in enumerate(frames_out):
            _frame = cv2.cvtColor(torch.clamp(frame, 0, 1).detach().permute(1, 2, 0).mul(255).cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR)
            _frame = cv2.putText(_frame, f"Compressed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _orig = cv2.cvtColor(torch.clamp(imgs[idx], 0, 1).detach().permute(1, 2, 0).mul(255).cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR)
            _orig = cv2.putText(_orig, f"Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _frame = cv2.hconcat([_orig, _frame])
            cv2.imwrite(f"/home/zlwang/ProtegoPlus/trash/vid_codec/vid_compressed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png", _frame)"""
        return torch.stack(frames_out, dim=0).to(imgs.device)

def occlude(imgs: torch.Tensor, occ_size: Union[float, Tuple[int, int]], size_mode: str = 'same_dim', occ_val: float = 0., occ_mode: str = 'center_bottom', differentiable: bool = False, sharpness: float = 20.0) -> torch.Tensor:
    """
    Occlude a region in the image

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        occ_size (Union[float, Tuple[int, int]]): Size of the occlusion (height, width) or a ratio
        size_mode (str): Mode for interpreting occ_size ('same_dim' or 'ratio')
            - 'same_dim': occ area same height and width as specified in occ_size. occ_size must be a tuple of (height, width)
            - 'same_size': occ area has the same size as specified in occ_size, but may have different height and width. occ_size must be a tuple of (height, width)
            - 'ratio': occ_size is a ratio of the image dimensions. occ_size must be a float in (0, 1)
        occ_val (float): Value to fill in the occluded region
        occ_mode (str): Mode for occlusion position ('center', 'center_bottom', 'random')
            - 'center': Occlude center of the image
            - 'center_bottom': Occlude center bottom of the image
            - 'random': Occlude a random position in the image
        differentiable (bool): Whether the occlusion should be differentiable (soft edges) or not (hard edges)
        sharpness (float): Sharpness of the occlusion edges if differentiable is True
    """
    B, C, H, W = imgs.shape
    occluded_imgs = imgs.clone()
    for i in range(B):
        if size_mode == 'same_dim':
            occ_h, occ_w = occ_size
        elif size_mode == 'same_size':
            size = occ_size[0] * occ_size[1]
            occ_h = random.choice(range(1, min(H, size)))
            occ_w = min(W, size // occ_h)
        elif size_mode == 'ratio':
            size = occ_size * H * W
            occ_h = random.choice(range(1, min(H, int(size))))
            occ_w = min(W, int(size) // occ_h)
        else:
            raise ValueError(f"Unsupported size_mode: {size_mode}")
        if occ_mode == 'center':
            cx = W // 2
            cy = H // 2
        elif occ_mode == 'center_bottom':
            cx = W // 2
            cy = H - occ_h // 2
        elif occ_mode == 'random':
            cx = random.randint(occ_w // 2, W - occ_w // 2)
            cy = random.randint(occ_h // 2, H - occ_h // 2)
        else:
            raise ValueError(f"Unsupported occ_mode: {occ_mode}")
        x1 = max(0, cx - occ_w // 2)
        x2 = min(W, cx + occ_w // 2)
        y1 = max(0, cy - occ_h // 2)
        y2 = min(H, cy + occ_h // 2)
        if not differentiable:
            occluded_imgs[i, :, y1:y2, x1:x2] = occ_val
        else:
            device = imgs.device
            y_coords = torch.arange(H, device=device, dtype=torch.float32)
            x_coords = torch.arange(W, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            mask_y = torch.sigmoid(sharpness * (yy - y1)) * torch.sigmoid(sharpness * (y2 - yy))
            mask_x = torch.sigmoid(sharpness * (xx - x1)) * torch.sigmoid(sharpness * (x2 - xx))
            soft_mask = mask_y * mask_x
            soft_mask = soft_mask.unsqueeze(0)
            occluded_imgs[i] = imgs[i] * (1 - soft_mask) + occ_val * soft_mask
    return occluded_imgs

def compress(imgs: torch.Tensor, method: str, **kwargs: Dict[str, Any]) -> torch.Tensor:
    """
    Compress images using specified method

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        method (str): Compression method ('gaussian', 'median', 'jpeg', 'resize', 'quantize', 'vid_codec', 'none')
        **kwargs: Additional parameters for the compression method

    Returns:
        Tensor: Compressed image tensor of shape [B, C, H, W]

    Raises:
        ValueError: If the specified method is not supported
    """
    method = method.lower().strip()
    if 'gaussian' in method:
        return gaussian_filter(imgs, **kwargs)
    elif 'median' in method:
        return median_filter(imgs, **kwargs)
    elif 'jpeg' in method:
        return jpeg(imgs, **kwargs)
    elif 'resize' in method:
        return resize(imgs, **kwargs)
    elif 'quantize' in method:
        return quantize(imgs, **kwargs)
    elif 'vid_codec' in method:
        return vid_compress(imgs, **kwargs)
    elif 'occlude' in method:
        return occlude(imgs, **kwargs)
    elif 'none' in method:
        return imgs
    else:
        raise ValueError(f"Unsupported compression method: {method}")
    