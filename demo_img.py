from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
# from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
h, w = 512, 960
# h, w = 480, 640
# cudnn.benchmark = True
def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 960, 512
    img = cv2.resize(img, (w, h))
    
    return img
def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
def draw_disparity(disparity_map):
    disparity_map = cv2.resize(disparity_map, (w, h))
    print(np.min(disparity_map), np.max(disparity_map))
    norm_disparity_map = 255 * ((disparity_map - np.min(disparity_map)) /
                                (np.max(disparity_map) - np.min(disparity_map)))

    return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_JET)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')
parser.add_argument('--model', default='CGI_Stereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--loadckpt', default='checkpoints/sceneflow/mid/checkpoint_000019.ckpt',help='load the weights from a specific checkpoint')
# parse arguments
args = parser.parse_args()


# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

###load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


save_dir = './test'

# test one sample
@make_nograd_func
def test_sample(left_img, right_img):
    model.eval()
    left_img = left_img.unsqueeze(0)
    right_img = right_img.unsqueeze(0)
    print(left_img.shape, right_img.shape)
    disp_ests = model(left_img.cuda(), right_img.cuda())
    return disp_ests[-1]

def test():
    os.makedirs(save_dir, exist_ok=True)
    left_img_path = r"111/im0.png"
    right_img_path = r"111/im1.png"
    left_img = load_image(left_img_path)
    right_img = load_image(right_img_path)
    processed = get_transform()
    left_img = processed(left_img)
    right_img = processed(right_img)
    disp_est_np = tensor2numpy(test_sample(left_img, right_img))
    torch.cuda.synchronize()
    disp_est_np = np.squeeze(disp_est_np)
    plt.imsave('disparity.png', -disp_est_np, cmap='jet')
    print(disp_est_np.shape)
    disparity_map = draw_disparity(disp_est_np)
    # cv2.imshow('disparity', disparity_map)
    # cv2.waitKey(0)


if __name__ == '__main__':
    test()
