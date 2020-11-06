import os
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from logo_scout.os2d.os2d.modeling.model import build_os2d_from_config
from logo_scout.os2d.os2d.config import cfg
import  logo_scout.os2d.os2d.utils.visualization as visualizer
from logo_scout.os2d.os2d.structures.feature_map import FeatureMapSize
from logo_scout.os2d.os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio

logger = setup_logger("OS2D")


cfg.is_cuda = torch.cuda.is_available()
cfg.init.model = "models/os2d_v2-train.pth"
net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)



def test_(input_image, class_images, class_ids):
  transform_image = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                        ])


  h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                                w=input_image.size[0],
                                                                target_size=1500)
  input_image = input_image.resize((w, h))

  input_image_th = transform_image(input_image)
  input_image_th = input_image_th.unsqueeze(0)
  if cfg.is_cuda:
      input_image_th = input_image_th.cuda()





  class_images_th = []
  for class_image in class_images:
      h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
                                                                w=class_image.size[0],
                                                                target_size=cfg.model.class_image_size)
      class_image = class_image.resize((w, h))

      class_image_th = transform_image(class_image)
      if cfg.is_cuda:
          class_image_th = class_image_th.cuda()

      class_images_th.append(class_image_th)



  with torch.no_grad():
      loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(images=input_image_th, class_images=class_images_th)




  image_loc_scores_pyramid = [loc_prediction_batch[0]]
  image_class_scores_pyramid = [class_prediction_batch[0]]
  img_size_pyramid = [FeatureMapSize(img=input_image_th)]
  transform_corners_pyramid = [transform_corners_batch[0]]




  boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                            img_size_pyramid, class_ids,
                                            nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                            nms_score_threshold=cfg.eval.nms_score_threshold,
                                            transform_corners_pyramid=transform_corners_pyramid)

  # remove some fields to lighten visualization                                       
  boxes.remove_field("default_boxes")




  # Note that the system outputs the correaltions that lie in the [-1, 1] segment as the detection scores (the higher the better the detection).
  scores = boxes.get_field("scores")





  # figsize = (8, 8)
  # fig=plt.figure(figsize=figsize)
  # columns = len(class_images)
  # for i, class_image in enumerate(class_images):
  #     fig.add_subplot(1, columns, i + 1)
  #     plt.imshow(class_image)
  #     plt.axis('off')






  # plt.rcParams["figure.figsize"] = figsize

  cfg.visualization.eval.max_detections = 8
  cfg.visualization.eval.score_threshold = float(0.6)
  show_detections(boxes, input_image,
                            cfg.visualization.eval)

  return boxes


import random
import string
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from logo_scout.os2d.os2d.structures.feature_map import FeatureMapSize
from logo_scout.os2d.os2d.structures.bounding_box import cat_boxlist, BoxList
from logo_scout.os2d.os2d.config import cfg


def show_detections(boxes, image_to_show,
                    cfg_visualization,
                    class_ids=None, image_id=None):
    labels = boxes.get_field("labels").clone()
    scores = boxes.get_field("scores").clone()

    if class_ids:
        for i_detection in range(labels.size(0)):
            labels[i_detection] = int(class_ids[labels[i_detection]])

    show_annotated_image(img=image_to_show,
                         boxes=boxes,
                         default_boxes=boxes.get_field("default_boxes") if boxes.has_field("default_boxes") else None,
                         transform_corners=boxes.get_field("transform_corners") if boxes.has_field(
                             "transform_corners") else None,
                         labels=labels,
                         scores=scores,
                         class_ids=class_ids,
                         score_threshold=cfg_visualization.score_threshold,
                         max_dets=cfg_visualization.max_detections,
                         showfig=True,
                         image_id=image_id)


def show_annotated_image(img, boxes, labels, scores, class_ids, score_threshold=0.0,
                         default_boxes=None, transform_corners=None,
                         max_dets=None, showfig=False, image_id=None):
    good_ids = torch.nonzero(scores.float() > score_threshold).view(-1)
    if good_ids.numel() > 0:
        if max_dets is not None:
            _, ids = scores[good_ids].sort(descending=False)
            good_ids = good_ids[ids[-max_dets:]]
        boxes = boxes[good_ids].cpu()
        labels = labels[good_ids].cpu()
        scores = scores[good_ids].cpu()
        label_names = ["Cl " + str(l.item()) for l in labels]
        box_colors = ["yellow"] * len(boxes)
    else:
        boxes = BoxList.create_empty(boxes.image_size)
        labels = torch.LongTensor(0)
        scores = torch.FloatTensor(0)
        label_names = []
        box_colors = []

    # create visualizations of default boxes
    if default_boxes is not None:
        default_boxes = default_boxes[good_ids].cpu()

        # append boxes
        boxes = torch.cat([default_boxes.bbox_xyxy, boxes.bbox_xyxy], 0)
        labels = torch.cat([torch.Tensor(len(default_boxes)).to(labels).zero_(), labels], 0)
        scores = torch.cat([torch.Tensor(len(default_boxes)).to(scores).fill_(float("nan")), scores], 0)
        label_names = [""] * len(default_boxes) + label_names
        box_colors = ["cyan"] * len(default_boxes) + box_colors
    else:
        boxes = boxes.bbox_xyxy

    if transform_corners is not None:
        # draw polygons representing the corners of a transformation
        transform_corners = transform_corners[good_ids].cpu()

    vis_image(img,
              showfig=showfig,
              boxes=boxes,
              scores=scores,
              label_names=label_names,
              colors=box_colors,
              image_id=image_id,
              polygons=transform_corners
              )
    return


def vis_image(img, boxes=None, label_names=None, scores=None, colors=None, image_id=None, polygons=None, showfig=False):
    """Visualize a color image.

    Args:
      img: (PIL.Image/tensor) image to visualize
      boxes: (tensor) bounding boxes, sized [#obj, 4], format: x1y1x2y2
      label_names: (list) label names
      scores: (list) confidence scores
      colors: (list) colors of boxes
      image_id: show this image_id as axes caption
      polygon: (tensor) quadrilateral defining the transformations [#obj, 8]
      showfig: (bool) - flag showing whether to call plt.show() at the end (e.g., stopping the script)

    Reference:
      https://github.com/kuangliu/torchcv/blob/master/torchcv/visualizations/vis_image.py
    """
    # Plot image
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if isinstance(img, torch.Tensor):
        img = torchvision.transforms.ToPILImage()(img.cpu())
    ax.imshow(img)

    # Plot boxes
    if boxes is not None:
        for i, bb in enumerate(boxes):

            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            new_logo = get_random_string(8)
            print(int(bb[0]), int(bb[1]), int(bb[0] + width), int(bb[1] + height))
            img.crop((int(bb[0]), int(bb[1]), int(bb[0] + width), int(bb[1] + height))).save(new_logo)

            box_color = 'red' if colors is None else colors[i]
            ax.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=box_color, linewidth=2))

            caption = []
            if label_names is not None:
                if label_names[i]:
                    try:
                        # if label_names is a pytorch vector
                        n = label_names[i].item()
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        # if scores is a list
                        n = label_names[i]

                    caption.append(str(n))

            if scores is not None:
                try:
                    # if scores is a pytorch vector
                    s = scores[i].item()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    # if scores is a list
                    s = scores[i]
                if not np.isnan(s):
                    caption.append('{:.4f}'.format(s))

            if len(caption) > 0:
                ax.text(bb[0], bb[1],
                        ': '.join(caption),
                        style='italic',
                        fontsize=8,
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2})

    # plot polygons in x1, y1, x2, y2, x3, y3, x4, y4 format
    if polygons is not None:
        for i, polygon in enumerate(polygons):
            xy = polygon.numpy()
            xy = xy.reshape((4, 2))
            xy = xy[[0, 2, 3, 1], :]
            ax.add_patch(plt.Polygon(
                xy, fill=False, edgecolor='red', linewidth=1))

    # Caption with image_id
    if image_id is not None:
        ax.set_title('Image {0}'.format(image_id))

    # turne off axes
    plt.axis('off')

    # Show
    if showfig:
        plt.show()

    return fig


import glob
import random

for _ in range(10):
  brand_name = random.choice( ["dream11"])
  print(brand_name)
  logo_paths = glob.glob("/content/drive/My Drive/BattleofBrands/Dataset/logos/predicted/"+brand_name+"/*.jpg")

  image_paths = glob.glob("/content/drive/My Drive/BattleofBrands/Dataset/SampleMatch/data_31/*.jpg")


  def get_random_string(length):
      letters = string.ascii_lowercase
      result_str = ''.join(random.choice(letters) for i in range(length))
      result_str = "/content/drive/My Drive/BattleofBrands/Dataset/logos/predicted/"+brand_name+"/"+result_str+".jpg"
      return result_str



  class_images = [read_image(random.choice(logo_paths)) for _ in range(10)]
  class_ids = [i for i in range(0, len(class_images))]

  count = 0
  for __ in range(250):
    image_path = random.choice(image_paths)
    count = count+1
    if count %50 ==0:
      print(count)
    input_image = read_image(image_path)
    test_(input_image, class_images, class_ids)