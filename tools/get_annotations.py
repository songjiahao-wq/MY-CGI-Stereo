import os
import glob
import os.path as osp
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

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/sceneflow/', dstype='frames_finalpass',
                 things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        # root = osp.join(self.root, 'FlyingThings3D')
        root = self.root
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:100])
        val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/left/*.png')))
        right_images = [image_file.replace('left', 'right') for image_file in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/*/left/*.png')))
        right_images = [image_file.replace('left', 'right') for image_file in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")



def generate_sceneflow_annotations():
    # 设置数据路径
    disparity_dir = r"D:\BaiduNetdiskDownload\SceneFlow\disparity"
    frames_dir = r"D:\BaiduNetdiskDownload\SceneFlow\frames_finalpass"
    output_file = "sceneflow_test.txt"

    # 获取所有场景文件夹
    scenes = os.listdir(frames_dir)
    
    with open(output_file, 'w') as f:
        for scene in scenes:
            # 获取左右图像路径
            left_img_pattern = os.path.join(frames_dir, scene, "left", "*.png")
            right_img_pattern = os.path.join(frames_dir, scene, "right", "*.png")
            
            # 获取视差图路径
            disp_pattern = os.path.join(disparity_dir, scene, "*.pfm")
            
            # 获取所有匹配的文件
            left_imgs = sorted(glob.glob(left_img_pattern))
            right_imgs = sorted(glob.glob(right_img_pattern))
            disp_imgs = sorted(glob.glob(disp_pattern))
            
            # 确保文件数量匹配
            assert len(left_imgs) == len(right_imgs) == len(disp_imgs), \
                f"文件数量不匹配: {scene}"
            
            # 写入文件
            for left, right, disp in zip(left_imgs, right_imgs, disp_imgs):
                line = f"{left} {right} {disp}\n"
                f.write(line)

if __name__ == "__main__":
    generate_sceneflow_annotations()
    print("训练文件生成完成！")
