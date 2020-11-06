import torch
import torchvision.transforms as transforms

from logo_scout.os2d.os2d.modeling.model import build_os2d_from_config
from logo_scout.os2d.os2d.config import cfg
from logo_scout.os2d.os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio

logger = setup_logger("OS2D")


cfg.is_cuda = torch.cuda.is_available()
cfg.init.model = "logo_scout/os2d/models/os2d_v2-train.pth"
net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)


class FewShotDetection:
    def __init__(self, logos_path):
        pass

    def identify_logos(self, image_path):
        pass
