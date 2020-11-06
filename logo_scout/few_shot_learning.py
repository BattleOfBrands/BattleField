import torch
import torchvision.transforms as transforms

from logo_scout.os2d.os2d.modeling.model import build_os2d_from_config
from logo_scout.os2d.os2d.config import cfg
from logo_scout.os2d.os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio


from logo_scout.os2d.os2d.structures.feature_map import FeatureMapSize
from logo_scout.os2d.os2d.structures.bounding_box import cat_boxlist, BoxList

logger = setup_logger("OS2D")


cfg.is_cuda = torch.cuda.is_available()
cfg.init.model = "logo_scout/os2d/models/os2d_v2-train.pth"
net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)


class FewShotDetection:
    def __init__(self, logos_path):
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(img_normalization["mean"], img_normalization["std"])
        ])
        self.logos = self.load_logos(logos_path)

    def transform_image(self, input_image, target_size):
        h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                                   w=input_image.size[0],
                                                                   target_size=target_size)
        input_image = input_image.resize((w, h))

        return self.transformer(input_image)

    def pre_process_input_image(self, image):
        input_image_th = self.transform_image(image, target_size=1500)
        input_image_th = input_image_th.unsqueeze(0)
        if cfg.is_cuda:
            input_image_th = input_image_th.cuda()
        return input_image_th


    def identify_logos(self, image_path):
        input_image_th = self.pre_process_input_image(image_path)
        with torch.no_grad():
            loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(
                images=input_image_th, class_images=self.logos)

        image_loc_scores_pyramid = [loc_prediction_batch[0]]
        image_class_scores_pyramid = [class_prediction_batch[0]]
        img_size_pyramid = [FeatureMapSize(img=input_image_th)]
        transform_corners_pyramid = [transform_corners_batch[0]]

        boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                         img_size_pyramid, [i for i in range(0, len(self.logos))],
                                         nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                         nms_score_threshold=cfg.eval.nms_score_threshold,
                                         transform_corners_pyramid=transform_corners_pyramid)

        # remove some fields to lighten visualization
        boxes.remove_field("default_boxes")
        return boxes


    def load_logos(self, logos_path):
        class_images_th = []
        for logos_path in logos_path:
            class_image = read_image(logo_path)
            class_image_th = self.transform_image(class_image, target_size=cfg.model.class_image_size)
            if cfg.is_cuda:
                class_image_th = class_image_th.cuda()

            class_images_th.append(class_image_th)
        return class_images_th



FewShotDetection()