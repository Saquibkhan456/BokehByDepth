import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from third_party.depthanything.depth_anything.dpt import DepthAnything
from third_party.depthanything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def estimate_depth(frame, encoder='vitl'):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    image = frame / 255.0
    
    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    return depth.detach().cpu()