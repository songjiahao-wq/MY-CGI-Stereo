import os

# 文件路径
txt_path = 'filenames/sceneflow_test.txt'
base_dir = 'D:/BaiduNetdiskDownload/SceneFlow/frames_finalpass/TRAIN'  # 修改为你的数据根目录路径（即包含 frames_finalpass 和 disparity 的上层目录）

# 读取并检查
with open(txt_path, 'r') as f:
    lines = f.readlines()

missing_files = []

for idx, line in enumerate(lines):
    paths = line.strip().split()
    if len(paths) != 3:
        print(f"[行 {idx+1}] 格式错误：{line}")
        continue
    
    full_paths = [os.path.join(base_dir, p) for p in paths]
    for p in full_paths:
        if not os.path.isfile(p):
            missing_files.append(p)
            print(f"缺失文件: {p}")

print(f"\n总共缺失文件数: {len(missing_files)}")
