import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    # 加载图像文件并将其转换为numpy数组
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # 将numpy数组转换为PyTorch张量，并调整维度顺序
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # 将图像张量添加一个维度并移动到指定设备
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    # 使用提供的参数初始化RAFT模型
    model = torch.nn.DataParallel(RAFT(args))
    # 从指定的检查点加载模型权重
    model.load_state_dict(torch.load(args.model))

    # 移除DataParallel包装器并将模型移动到指定设备
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # 获取指定路径中所有图像文件的列表
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        # 对图像进行排序以确保正确配对
        images = sorted(images)
        # 迭代成对的连续图像
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            # 加载两张图像
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # 填充图像以确保它们具有相同的尺寸
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # 计算两张图像之间的光流
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # 可视化光流
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
