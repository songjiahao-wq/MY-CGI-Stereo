# -*- coding: utf-8 -*-
# @Time    : 2025/11/6 上午10:30
# @Author  : sjh
# @Site    : 
# @File    : 111.py
# @Comment :
import timm

# 查看全部模型（上千个）
all_models = timm.list_models()
print(len(all_models))
# print(all_models)  # 打印前50个
mobilenet_models = timm.list_models('*mobilenet*')
for name in mobilenet_models:
    print(name)