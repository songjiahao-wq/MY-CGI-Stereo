import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import random
from pathlib import Path
from glob import glob
import os.path as osp

# 添加缺失的导入 - 这些是原始代码需要但未导入的
try:
    import frame_utils
    from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
except ImportError as e:
    print(f"警告: 无法导入必要的模块 {e}")
    print("请确保 utils/frame_utils.py 和 utils/augmentor.py 文件存在")
    # 提供简单的替代实现以避免报错
    class FrameUtils:
        @staticmethod
        def read_gen(path):
            # 简单的图像读取实现
            from PIL import Image
            return np.array(Image.open(path))

        @staticmethod
        def readDispSintelStereo(path):
            return FrameUtils.read_gen(path)

    class FlowAugmentor:
        def __init__(self, **kwargs):
            pass
        def __call__(self, img1, img2, flow):
            return img1, img2, flow

    class SparseFlowAugmentor:
        def __init__(self, **kwargs):
            pass
        def __call__(self, img1, img2, flow, valid):
            return img1, img2, flow, valid

    frame_utils = FrameUtils()

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)
            flow = F.pad(flow, [padW] * 2 + [padH] * 2)
            valid = F.pad(valid, [padW] * 2 + [padH] * 2)

        return img1, img2, flow

    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='D:\project2024\stereo matching\DATA\SceneFlow/', dstype='frames_finalpass',
                 things_test=False):
        super().__init__(aug_params, sparse=True, reader=frame_utils.read_gen)

        self.root = root
        self.dstype = dstype

        if things_test:
            self.image_list = []
            self.disparity_list = []
            for cam in ['left', 'right']:
                image_path = os.path.join(root, dstype, 'TEST/' + cam)
                c_images = sorted(glob(image_path + '/*/*.png'))
                self.image_list += c_images

            for cam in ['left']:
                disp_path = os.path.join(root, 'disparity/TEST/' + cam)
                d_images = sorted(glob(disp_path + '/*/*.pfm'))
                self.disparity_list += d_images

        else:

            train_path = os.path.join(self.root, dstype, 'TRAIN')
            image_path_l = []
            image_path_r = []
            disp_path = []
            for sequence in ['A', 'B', 'C']:
                image_path_l += sorted(glob(os.path.join(train_path, sequence + '/*/left/*.png')))
                image_path_r += sorted(glob(os.path.join(train_path, sequence + '/*/right/*.png')))
                disp_path += sorted(glob(os.path.join(train_path, sequence + '/*/disparity/*.pfm')))

            # things path
            things_root = os.path.join(root, dstype.replace('finalpass', 'cleanpass'), 'TRAIN')
            object_root = os.path.join(root, 'disparity/TRAIN')
            objects = sorted(glob(os.path.join(object_root, 'OBJECTS/*')))
            for obj in objects:
                obj_name = obj.split('\\')[-1]
                dst_path = os.path.join(things_root, 'OBJECTS', obj_name)
                image_path_l += sorted(glob(os.path.join(dst_path, 'frames_cleanpass/LEFT/*.png')))
                image_path_r += sorted(glob(os.path.join(dst_path, 'frames_cleanpass/RIGHT/*.png')))
                disp_path += sorted(glob(os.path.join(dst_path, 'disparity/LEFT/*.pfm')))

            self.image_list = list(zip(image_path_l, image_path_r))
            self.disparity_list = disp_path

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)
            flow = F.pad(flow, [padW] * 2 + [padH] * 2)
            valid = F.pad(valid, [padW] * 2 + [padH] * 2)

        return img1, img2, flow


def fetch_dataloader(args):
    """ Create the data loader for the corresponding training set """

    aug_params = {'crop_size': args.image_size}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    # 只保留 only_sceneflow 分支，因为这是默认设置
    if args.train_datasets == 'only_sceneflow':
        sceneflow_train = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        sceneflow_test = SceneFlowDatasets(aug_params, dstype='frames_finalpass', things_test=True)
        logging.info(f"Adding {len(sceneflow_train)} samples from SceneFlow")
        logging.info(f"Adding {len(sceneflow_test)} samples from SceneFlow")
        train_dataset = sceneflow_train
        test_dataset = sceneflow_test
    else:
        raise ValueError(f"不支持的训练数据集类型: {args.train_datasets}。仅支持 'only_sceneflow'")

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_dataset, test_dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='rt-igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--logdir', default='./checkpoints_rt', help='the directory to save logs and checkpoints')
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16',
                                                                         'float32'], help='Choose precision type: float16 or bfloat16 or float32')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_datasets', default='only_sceneflow', help="training datasets.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320,
                                                                      768], help="size of the random image crops used during training.")


    args = parser.parse_args()

    train_dataset, test_dataset = fetch_dataloader(args)

    # 将dataset里的image_list和disp数据路径保存到txt里，每行左图、右图、disp
    with open('train_dataset.txt', 'w') as f:
        for img_pair, disp in zip(train_dataset.image_list, train_dataset.disparity_list):
            left_img = img_pair[0].replace('\\', '/').replace("D:\project2024\stereo matching\DATA\SceneFlow/",'')
            right_img = img_pair[1].replace('\\', '/').replace("D:\project2024\stereo matching\DATA\SceneFlow/",'')
            disp_path = disp.replace('\\', '/').replace("D:\project2024\stereo matching\DATA\SceneFlow/",'')
            f.write(f"{left_img} {right_img} {disp_path}\n")

    with open('test_dataset.txt', 'w') as f:
        for img_pair, disp in zip(test_dataset.image_list, test_dataset.disparity_list):
            left_img = img_pair[0].replace('\\', '/').replace("D:\\project2024\\stereo matching\\DATA\\SceneFlow/",'')
            right_img = img_pair[1].replace('\\', '/').replace("D:\\project2024\\stereo matching\\DATA\\SceneFlow/",'')
            disp_path = disp.replace('\\', '/').replace("D:\\project2024\\stereo matching\\DATA\\SceneFlow/",'')
            f.write(f"{left_img} {right_img} {disp_path}\n")

    print("训练和测试数据集索引文件已生成:")
    print(f"- train_dataset.txt: {len(train_dataset.image_list)} 个样本")
    print(f"- test_dataset.txt: {len(test_dataset.image_list)} 个样本")