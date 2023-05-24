#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import cv2
import os
import sys
import os.path as osp
import glob
import padding
import copy
from PIL import Image
import time

import numpy as np
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer


def get_args_parser_top(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Inference.', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/yolov6m6.pt', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')
    parser.add_argument('--webcam', action='store_true', help='whether to use webcam.')
    parser.add_argument('--webcam-addr', type=str, default='0', help='the web camera address, local camera or rtsp address.')
    parser.add_argument('--yaml', type=str, default='data/coco.yaml', help='data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1080,1920], help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=0.07, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.')
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt',default=True,  action='store_true', help='save results to *.txt.')
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')
    parser.add_argument('--view-img', action='store_true', help='show inference results')
    parser.add_argument('--classes',default=[39,76] ,nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--project', default='runs/inference_top', help='save inference results to project/name.')
    parser.add_argument('--name', default='exp', help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')
    # default=[39,79,76],

    args = parser.parse_args()
    LOGGER.info(args)
    return args

def get_args_parser_middle(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Inference.', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/yolov6m6.pt', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')
    parser.add_argument('--webcam', action='store_true', help='whether to use webcam.')
    parser.add_argument('--webcam-addr', type=str, default='0', help='the web camera address, local camera or rtsp address.')
    parser.add_argument('--yaml', type=str, default='data/coco.yaml', help='data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1080,1920], help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=0.17, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.')
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt.')
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')
    parser.add_argument('--view-img', action='store_true', help='show inference results')
    parser.add_argument('--classes', default=[39,76],nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--project', default='runs/inference_middle', help='save inference results to project/name.')
    parser.add_argument('--name', default='exp', help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')

    args = parser.parse_args()
    LOGGER.info(args)
    return args

def get_args_parser_bottom(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Inference.', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/yolov6m6.pt', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')
    parser.add_argument('--webcam', action='store_true', help='whether to use webcam.')
    parser.add_argument('--webcam-addr', type=str, default='0', help='the web camera address, local camera or rtsp address.')
    parser.add_argument('--yaml', type=str, default='data/coco.yaml', help='data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1080,1920], help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.')
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt.')
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')
    parser.add_argument('--view-img', action='store_true', help='show inference results')
    parser.add_argument('--classes',default=[39,76], nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--project', default='runs/inference_bottom', help='save inference results to project/name.')
    parser.add_argument('--name', default='exp', help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')

    args = parser.parse_args()
    LOGGER.info(args)
    return args

@torch.no_grad()
def run(weights=osp.join(ROOT, 'yolov6s.pt'),
        source=osp.join(ROOT, 'data/images'),
        webcam=False,
        webcam_addr=0,
        yaml=None,
        img_size=640,
        conf_thres=0.4,
        iou_thres=0.45,
        max_det=1000,
        device='',
        save_txt=False,
        not_save_img=False,
        save_dir=None,
        view_img=True,
        classes=None,
        agnostic_nms=False,
        project=osp.join(ROOT, 'runs/inference'),
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False,
        imgs = []
        ):
    """ Inference process, supporting inference on one image file or directory which containing images.
    Args:
        weights: The path of model.pt, e.g. yolov6s.pt
        source: Source path, supporting image files or dirs containing images.
        yaml: Data yaml file, .
        img_size: Inference image-size, e.g. 640
        conf_thres: Confidence threshold in inference, e.g. 0.25
        iou_thres: NMS IOU threshold in inference, e.g. 0.45
        max_det: Maximal detections per image, e.g. 1000
        device: Cuda device, e.e. 0, or 0,1,2,3 or cpu
        save_txt: Save results to *.txt
        not_save_img: Do not save visualized inference results
        classes: Filter by class: --class 0, or --class 0 2 3
        agnostic_nms: Class-agnostic NMS
        project: Save results to project/name
        name: Save results to project/name, e.g. 'exp'
        line_thickness: Bounding box thickness (pixels), e.g. 3
        hide_labels: Hide labels, e.g. False
        hide_conf: Hide confidences
        half: Use FP16 half-precision inference, e.g. False
    """
    # create save dir
    if save_dir is None:
        save_dir = osp.join(project, name)
        save_txt_path = osp.join(save_dir, 'labels')
    else:
        save_txt_path = save_dir
    if (not not_save_img or save_txt) and not osp.exists(save_dir):
        os.makedirs(save_dir)
    else:
        LOGGER.warning('Save directory already existed')
    if save_txt:
        save_txt_path = osp.join(save_dir, 'labels')
        if not osp.exists(save_txt_path):
            os.makedirs(save_txt_path)

    # Inference
    # start_time = time.time()  
    inferer = Inferer(source, webcam, webcam_addr, weights, device, yaml, img_size, half, imgs)
    # end_time = time.time()
        # 计算执行时间
    # execution_time = end_time - start_time
    
    # 输出执行时间
    # print(f"inferer程序执行时间为: {execution_time} 秒")
    # start_time = time.time() 
    img_maps = inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, not not_save_img, hide_labels, hide_conf, view_img)
    # print(img_maps)
    # end_time = time.time()
        # 计算执行时间
    # execution_time = end_time - start_time
    
    # 输出执行时间
    # print(f"img_maps程序执行时间为: {execution_time} 秒")


    crop_images_list = []
    for image_tensor,img_map in zip(imgs,img_maps):
        crop_images = []
        # print(len(img_map))
        # 获取坐标和长宽
        # img_map = np.expand_dims(img_map, axis=1) 
        imgs_info = np.array(img_map)
        
        # print(image_tensor.shape)
        # print(imgs_info.shape)
        # print(type(img_map))
        # x, y, w, h = img_map[:, :4].unbind(dim=1)
        x = imgs_info[:,0]
        y = imgs_info[:,1]
        w = imgs_info[:,2]
        h = imgs_info[:,3]
        # print(x,y,w,h)
        img_h, img_w, _ = image_tensor.shape
        # print(img_h)
        x=x*img_w
        w=w*img_w
        y=y*img_h
        h=h*img_h


        # 计算边界框坐标
        x1 = (x - w / 2) 
        y1 = (y - h / 2)
        x2 = (x + w / 2) 
        y2 = (y + h / 2) 
        x1 = x1.astype('int64')
        
        y1 = y1.astype('int64')
        
        x2 = x2.astype('int64')
        
        y2 = y2.astype('int64')

        # 将坐标限制在图像范围内
        # print(image_tensor.shape[0])
        # print(image_tensor.shape[1])
        # x1 = torch.clamp(x1, min=0, max=image_tensor.shape[1])
        # y1 = torch.clamp(y1, min=0, max=image_tensor.shape[0])
        # x2 = torch.clamp(x2, min=0, max=image_tensor.shape[1])
        # y2 = torch.clamp(y2, min=0, max=image_tensor.shape[0])
        # 将坐标转换为整数类型


        # print(img_map.shape[0])
        for i in range(len(img_map)):
            # print(image_tensor)
            # # 是w,h,c
            crop_image = image_tensor[y1[i]:y2[i],x1[i]:x2[i]]
            # print(image_tensor.shape)
            crop_images.append(crop_image)
        # print(len(crop_images))
        # 裁剪后的图像张量添加到总列表
        crop_images_list.append(crop_images)

    # print(len(crop_images_list))
# 保存检查裁剪正确性    
    # for i, crop_images in enumerate(crop_images_list):
    #     for j, crop_image in enumerate(crop_images):

            
    # #         # 将BGR通道顺序转换为RGB通道顺序
    #         rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    #         pil_image = Image.fromarray(rgb_image)

    # #         # 保存图像
    #         pil_image.save(f'cropped_image_{i}_{j}.jpg')
    #         # print(crop_image.shape)
            # print(rgb_image.shape)
            # 保存图像
            # cv2.imwrite(f'cropped_image_{i}_{j}.jpg', rgb_image)

    # print(crop_images_list)

    # print(imgs)
    # for i in imgs:
    #     print(img_map[i].shape)
    # print(img_map.shape[0])

    # crop_images = []
    # for i in len(img_map.shape[0]):
    #     crop_image = imgs[y1[i]:y2[i], x1[i]:x2[i]]
    #     crop_images.append(crop_image)


    if save_txt or not not_save_img:
        LOGGER.info(f"Results saved to {save_dir}")




src_folder = 'data/images/'
png_files = glob.glob(os.path.join(src_folder, '*.png'))

if __name__ == "__main__":
 
    # top = []
    # middle = []
    # bottom = []
    imgs = []
    for png_file in png_files:
        img = cv2.imread(png_file)

        # img_top=padding.padding_top(img=img,top_percent=0.15,bottom_percent=0.7,left_percent=0.0,right_percent=0.16)
        # # print(img_top.shape)
        # # print(img)
        # img_middle=padding.padding_middle(img=img,top_percent=0.30,bottom_percent=0.39,left_percent=0.0,right_percent=0.18)
        # print(img_middle)
        # img_bottom=padding.padding_bottom(img=img,top_percent=0.59,bottom_percent=0.15,left_percent=0.0,right_percent=0.18)
        # cv2.imwrite('top.jpg', img_top)
        imgs.append(img)
        # middle.append(img_middle)
        # bottom.append(img_bottom)
    # print(len(top))
    # print(middle)
    start_time = time.time()  
    args_top = get_args_parser_top()
    run(**vars(args_top), imgs = imgs)
    end_time = time.time()
    #     # 计算执行时间
    execution_time = end_time - start_time
    
    # args_middle = get_args_parser_middle()
    # run(**vars(args_middle), imgs = middle)

    # args_bottom = get_args_parser_bottom()
    # run(**vars(args_bottom), imgs = bottom)


    # # 输出执行时间
    print(f"程序执行时间为: {execution_time} 秒")
   