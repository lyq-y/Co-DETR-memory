import torch
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 假设这个是提取patch的函数实现
def extract_drone_patches_with_bbox(images, labels, patch_size=16):
    """
    根据标签中的边界框提取包含Drone的patch。

    Args:
        images (Tensor): 输入的图像数据，形状为 [B, C, H, W]
        labels (List[List[Tuple[int, int, int, int]]]): 每张图的目标边界框列表，
            格式为 [(xmin, ymin, w, h), ...]，每个图像的标签是一个边界框的列表。
        patch_size (int): 提取的patch大小。

    Returns:
        List[Tuple[Tensor, Tuple[int, int]]]: 含有目标Drone的patch列表，以及每个patch在原图中的位置
    """
    drone_patches = []

    for img, bbox_list in zip(images, labels):
        h_img, w_img = img.shape[1:]  # 获取图像尺寸

        # 遍历每个边界框（每张图可能有多个目标Drone）
        for bbox in bbox_list:
            xmin, ymin, w, h = bbox
            xmax, ymax = xmin + w, ymin + h

            # 遍历每个patch
            for i in range(0, h_img - patch_size + 1, patch_size):
                for j in range(0, w_img - patch_size + 1, patch_size):
                    # 计算当前patch的范围
                    patch_x_min = j
                    patch_y_min = i
                    patch_x_max = j + patch_size
                    patch_y_max = i + patch_size

                    # 判断是否与目标范围有交集
                    if not (patch_x_max <= xmin or patch_x_min >= xmax or
                            patch_y_max <= ymin or patch_y_min >= ymax):
                        # 如果有交集，将patch和其位置保存
                        patch = img[:, i:i+patch_size, j:j+patch_size]
                        drone_patches.append((patch, (patch_x_min, patch_y_min)))

    return drone_patches


# 主程序
def extract_and_save_patches(image_path, bbox, patch_size, output_dir):
    """
    从给定的图像和边界框中提取交集的patch，并保存为图片。

    Args:
        image_path (str): 输入图像的路径。
        bbox (List[int]): 边界框坐标 [xmin, ymin, w, h]。
        patch_size (int): 提取的patch的大小。
        output_dir (str): 输出保存patch的目录。
    """
    # 加载图像并转为Tensor
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)  # 转为 [1, C, H, W]

    # 计算patches
    labels = [[tuple(bbox)]]  # 包装成符合函数输入格式的标签

    # 提取patch
    drone_patches = extract_drone_patches_with_bbox(img_tensor, labels, patch_size)

    # 保存所有的patch
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (patch, (patch_x_min, patch_y_min)) in enumerate(drone_patches):
        # 打印patch的位置
        print(f"Saving patch {idx}: Position (x, y) = ({patch_x_min}, {patch_y_min})")

        # 转换为 H x W x C 格式并保存
        patch_np = patch.permute(1, 2, 0).numpy()  # 转回为 H x W x C 格式
        patch_img = Image.fromarray(np.uint8(patch_np))  # 转为PIL Image
        patch_img.save(os.path.join(output_dir, f"patch_{idx}.png"))

    # 可视化原图和边界框
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    xmin, ymin, w, h = bbox
    xmax, ymax = xmin + w, ymin + h
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='r', facecolor='none'))
    plt.title("Original Image with BBox")
    output_path = 'path_to_save_image.jpg'  # 指定保存图像的路径和文件名
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # 保存图像，删除多余空白

    plt.show()

# 测试提取和保存patch
image_path = '/home/liuyuqing/liuyuqing/dataset/ARD-MAV/Coco-ARD/val/phantom02_1194.jpg'
bbox = [102.0, 287.0, 20.0, 14.0]  # xmin, ymin, width, height
patch_size = 16  # 假设patch大小为16x16
output_dir = './output_patches'  # 设置输出目录

extract_and_save_patches(image_path, bbox, patch_size, output_dir)
