import numpy as np
import re
import matplotlib.pyplot as plt

def read_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'  # little or big endian
        scale = abs(scale)

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM files are stored in bottom-up order

        return data
import os
import cv2
def show_disparity(pfm_path):
    color_path = pfm_path.replace('.pfm', '.png').replace('disparity', 'frames_finalpass')
    color_img = cv2.imread(color_path)
    
    disp = read_pfm(pfm_path)
    # disp = disp.astype(np.float32)
    disp = -disp
    print(np.min(disp), np.max(disp))
    print(disp.shape, color_img.shape)
    # 将无效区域（负值）遮掉
    valid_mask = disp > 0
    disp_vis = np.copy(disp)
    disp_vis[~valid_mask] = 0
    
    # 可视化color_img和disp_vis
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(color_img)
    plt.title("Color Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(disp_vis, cmap='plasma')
    plt.title("Disparity Map (PFM)")
    plt.colorbar(label='disparity (pixels)')
    plt.axis('off')
    plt.show()
pfm_path = 'D:/BaiduNetdiskDownload/SceneFlow/disparity/TEST/left'
# pfm_path = r"D:\BaiduNetdiskDownload\SceneFlow\disparity\TRAIN\35mm_focallength\scene_backwards\fast\left"
# pfm_path = r"D:\BaiduNetdiskDownload\SceneFlow\disparity\TRAIN\15mm_focallength\scene_forwards\slow\left"
# pfm_path = 'D:/BaiduNetdiskDownload/SceneFlow/disparity/TRAIN/funnyworld_augmented1_x2/left/'
pfm_file = os.path.join(pfm_path, '0000000.pfm')

show_disparity(pfm_file)
