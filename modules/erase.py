import concurrent.futures
import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from modules.sttn import build_sttn_model, inpaint_video_with_builded_sttn
from utils.image_utils import load_img


@torch.no_grad()
def inpaint_video(
    paths_list: List[str],
    frames_list: List[Image.Image],
    masks_list: List[Image.Image],
    neighbor_stride: int,
    ckpt_p="./models/sttn.pth",
):
    """
    对视频帧进行修复。

    使用预训练的 STTN 模型对视频帧进行修复。根据设备情况选择在 CUDA 或 CPU 上执行修复过程。此函数处理每个视频的帧序列，尝试恢复帧中的缺失部分。

    参数:
    - paths_list: 视频帧路径列表。
    - frames_list: 视频帧图像列表。
    - masks_list: 帧掩码图像列表。
    - neighbor_stride: 邻居帧之间的步长。
    - ckpt_p: STTN 模型检查点文件路径。

    返回:
    - 修复后的视频帧图像路径列表。
    """
    # 构建模型时获取设备信息
    model, device = build_sttn_model(ckpt_p)

    results = []
    for paths, frames, masks in tqdm(
        zip(paths_list, frames_list, masks_list),
        desc="Inpaint job",
        total=len(paths_list),
    ):
        # 使用获取到的设备
        result = inpaint_video_with_builded_sttn(
            model, paths, frames, masks, neighbor_stride, device
        )
        results.extend(result)

    return results


def inpaint_imag(mask_result: List[tuple]):
    """
    对掩码处理后的图像进行修复。

    使用多线程并行处理的方法，对每个掩码帧进行处理并保存图像。
    这里使用了tqdm来显示处理进度，使程序在执行时能给出进度反馈。

    参数:
    mask_result: 掩码处理后的结果，是一个包含多个帧的可迭代对象。

    返回:
    None
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_frame, mask_result),
                total=len(mask_result),
                desc="Save Image",
            )
        )
    return None


def process_frame(value: tuple):
    """
    处理并保存单个视频帧。

    参数:
    value: 一个元组，第一个元素是帧的文件路径，第二个元素是帧的图像数据（以计算生成的方式获得）。

    返回值:
    无返回值，但该函数会直接在文件系统中生成视频帧图像文件。
    """
    frame_path, comp_frame = value
    Image.fromarray(np.uint8(comp_frame)).save(frame_path)


def extract_mask(
    mask_result,
    fps,
    frame_len,
    max_frame_length,
    min_frame_length,
    mask_expand: int = 30,
):
    """修改提取掩码的逻辑，更好地处理短字幕"""
    
    def calculate_dynamic_mask_expand(box_height):
        return max(mask_expand, int(box_height * 0.5))
    
    paths_list = []
    frames_list = []
    masks_list = []
    paths = []
    frames = []
    masks = []
    frame_number_pre = 0
    
    # 按帧号排序处理
    sorted_frames = sorted(mask_result.items(), key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
    
    for frame_path, value in tqdm(sorted_frames, desc="Find Mask"):
        frame_number = int(os.path.splitext(os.path.basename(frame_path))[0])
        image = load_img(frame_path)
        width_ = image.size[0]
        mask = np.zeros(image.size[::-1], dtype="uint8")
        xmin, ymin, xmax, ymax = value["box"]
        
        # 修正掩码区域计算
        box_height = ymax - ymin
        dynamic_mask_expand = calculate_dynamic_mask_expand(box_height)
        
        # 直接使用检测到的坐标，不需要计算xwidth
        cv2.rectangle(
            mask,
            (max(0, xmin - dynamic_mask_expand), max(0, ymin - dynamic_mask_expand)),
            (min(xmax + dynamic_mask_expand, width_ - 1), min(ymax + dynamic_mask_expand, image.size[1] - 1)),
            (255, 255, 255),
            thickness=-1,
        )
        mask = Image.fromarray(mask)
        
        # 修改帧组合逻辑
        if len(paths) == 0:
            paths = [frame_path]
            frames = [image]
            masks = [mask]
        elif frame_number - frame_number_pre <= int(fps * 1.5):  # 增加容忍度
            paths.append(frame_path)
            frames.append(image)
            masks.append(mask)
        else:
            # 保存当前组并开始新组
            if len(paths) >= max(2, int(fps * 0.1)):  # 允许更短的字幕组，但至少需要2帧
                paths_list.append(paths)
                frames_list.append(frames)
                masks_list.append(masks)
            paths = [frame_path]
            frames = [image]
            masks = [mask]
        
        frame_number_pre = frame_number
    
    # 处理最后一组
    if len(paths) >= max(2, int(fps * 0.1)):
        paths_list.append(paths)
        frames_list.append(frames)
        masks_list.append(masks)
    
    return paths_list, frames_list, masks_list


def remove_subtitles(ocr_result, fps, total_frames, config):
    """
    移除视频中的字幕。

    参数:
    - ocr_result: dict, OCR 识别的结果，包含需要移除的字幕信息。
    - fps: float, 视频的帧率，用于计算视频处理的速度。
    - frame_len: int, 帧的长度，用于调整视频处理的精度。
    - config: dict, 配置文件，包含视频处理的参数。

    返回值:
    无。
    """
    # 扩大擦除区域
    def expand_box(box, margin=10):
        x1, y1, x2, y2 = box
        return [
            max(0, x1 - margin),
            max(0, y1 - margin),
            x2 + margin,
            y2 + margin
        ]
    
    # 对检测到的区域进行扩展处理
    for frame_idx in ocr_result:
        if 'box' in ocr_result[frame_idx]:
            ocr_result[frame_idx]['box'] = expand_box(ocr_result[frame_idx]['box'])
    
    paths_list, frames_list, masks_list = extract_mask(
        ocr_result,
        fps,
        total_frames,
        config["erase"]["max_frame_length"],
        config["erase"]["min_frame_length"],
        config["erase"]["mask_expand"],
    )
    results = inpaint_video(
        paths_list,
        frames_list,
        masks_list,
        config["erase"]["neighbor_stride"],
        config["erase"]["ckpt_p"],
    )
    inpaint_imag(results)


def debug_subtitle_erase(
    test_frames: list,
    ocr_result: dict,
    fps: float,
    config: dict,
    output_dir: str = "debug_output"
):
    """
    调试字幕擦除流程

    参数:
    test_frames: list - 要测试的帧号列表，例如[473, 474, ..., 498]
    ocr_result: dict - OCR检测结果
    fps: float - 视频帧率
    config: dict - 配置信息
    output_dir: str - 调试输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def save_debug_image(img, stage, frame_num):
        """保存调试图像"""
        debug_path = os.path.join(output_dir, f"{frame_num:04d}_{stage}.png")
        if isinstance(img, np.ndarray):
            cv2.imwrite(debug_path, img)
        elif isinstance(img, Image.Image):
            img.save(debug_path)
            
    def visualize_box(image, box, color=(0, 255, 0), thickness=2):
        """在图像上可视化边界框"""
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        return image

    # 1. 检查原始图像和OCR检测框
    print("1. 检查OCR检测结果...")
    for frame_num in test_frames:
        frame_path = f"3/{frame_num:04d}.png"
        if frame_path + ",0" not in ocr_result:
            print(f"警告: 帧 {frame_num} 未找到OCR结果")
            continue
            
        # 读取原始图像
        original_img = cv2.imread(frame_path)
        if original_img is None:
            print(f"错误: 无法读取帧 {frame_num}")
            continue
            
        # 绘制OCR检测框
        box = ocr_result[frame_path + ",0"]["box"]
        vis_img = original_img.copy()
        vis_img = visualize_box(vis_img, box)
        save_debug_image(vis_img, "1_ocr_box", frame_num)

    # 2. 检查mask生成
    print("\n2. 检查mask生成...")
    mask_expand = config["erase"]["mask_expand"]
    for frame_num in test_frames:
        frame_path = f"3/{frame_num:04d}.png"
        if frame_path + ",0" not in ocr_result:
            continue
            
        # 读取图像
        image = Image.open(frame_path)
        width_, height_ = image.size
        
        # 生成mask
        mask = np.zeros((height_, width_), dtype="uint8")
        box = ocr_result[frame_path + ",0"]["box"]
        xmin, ymin, xmax, ymax = box
        
        # 计算动态扩展值
        box_height = ymax - ymin
        dynamic_mask_expand = max(mask_expand, int(box_height * 0.5))
        
        # 绘制mask
        cv2.rectangle(
            mask,
            (max(0, xmin - dynamic_mask_expand), max(0, ymin - dynamic_mask_expand)),
            (min(xmax + dynamic_mask_expand, width_ - 1), min(ymax + dynamic_mask_expand, height_ - 1)),
            255,
            thickness=-1,
        )
        
        # 保存mask可视化结果
        save_debug_image(mask, "2_mask", frame_num)
        
        # 将mask应用到原图上以检查覆盖区域
        original_img = cv2.imread(frame_path)
        masked_img = original_img.copy()
        masked_img[mask > 0] = [0, 0, 255]  # 用红色显示mask区域
        save_debug_image(masked_img, "2_masked", frame_num)

    # 3. 检查STTN输入
    print("\n3. 检查STTN模型输入...")
    w, h = 432, 240  # STTN的处理尺寸
    for frame_num in test_frames:
        frame_path = f"3/{frame_num:04d}.png"
        if frame_path + ",0" not in ocr_result:
            continue
            
        # 读取并缩放图像
        image = Image.open(frame_path)
        resized_img = image.resize((w, h))
        save_debug_image(resized_img, "3_resized", frame_num)
        
        # 生成缩放后的mask
        mask = Image.fromarray(np.zeros(image.size[::-1], dtype="uint8"))
        box = ocr_result[frame_path + ",0"]["box"]
        xmin, ymin, xmax, ymax = box
        box_height = ymax - ymin
        dynamic_mask_expand = max(mask_expand, int(box_height * 0.5))
        
        mask_np = np.array(mask)
        cv2.rectangle(
            mask_np,
            (max(0, xmin - dynamic_mask_expand), max(0, ymin - dynamic_mask_expand)),
            (min(xmax + dynamic_mask_expand, width_ - 1), min(ymax + dynamic_mask_expand, height_ - 1)),
            255,
            thickness=-1,
        )
        mask = Image.fromarray(mask_np)
        resized_mask = mask.resize((w, h), Image.NEAREST)
        save_debug_image(resized_mask, "3_resized_mask", frame_num)

    print("\n调试信息已保存到", output_dir)
    print("请检查以下内容：")
    print("1. *_1_ocr_box.png - OCR检测框是否准确覆盖字幕")
    print("2. *_2_mask.png - 生成的mask是否完整覆盖字幕区域")
    print("2. *_2_masked.png - mask覆盖的原图区域是否正确")
    print("3. *_3_resized.png 和 *_3_resized_mask.png - STTN输入是否正确")


if __name__ == "__main__":
    import json
    import argparse
    
    def load_test_data():
        """加载测试数据"""
        # 测试OCR结果
        test_ocr_result = {}
        for i in range(473, 499):
            frame_key = f"3/{i:04d}.png,0"
            test_ocr_result[frame_key] = {
                "box": [451, 1418, 623, 1518],
                "text": "什么"
            }
        return test_ocr_result

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='调试字幕擦除')
    parser.add_argument('--fps', type=float, default=30.0, help='视频帧率')
    parser.add_argument('--output_dir', type=str, default='debug_output', help='调试输出目录')
    parser.add_argument('--ocr_path', type=str, help='OCR结果JSON文件路径（可选）')
    args = parser.parse_args()

    # 配置信息
    config = {
        "erase": {
            "mask_expand": 30,
            "max_frame_length": 100,
            "min_frame_length": 2,
            "neighbor_stride": 10,
            "ckpt_p": "./models/sttn.pth"
        }
    }

    # 加载OCR结果
    if args.ocr_path and os.path.exists(args.ocr_path):
        print(f"从文件加载OCR结果: {args.ocr_path}")
        with open(args.ocr_path, 'r', encoding='utf-8') as f:
            ocr_result = json.load(f)
    else:
        print("使用测试OCR结果")
        ocr_result = load_test_data()

    # 测试帧范围
    test_frames = list(range(473, 499))

    print("\n=== 开始调试字幕擦除 ===")
    print(f"测试帧范围: {test_frames[0]} - {test_frames[-1]}")
    print(f"输出目录: {args.output_dir}")
    
    # 1. 调试OCR结果和mask生成
    debug_subtitle_erase(
        test_frames=test_frames,
        ocr_result=ocr_result,
        fps=args.fps,
        config=config,
        output_dir=args.output_dir
    )
    
    # 2. 测试完整的擦除流程
    print("\n=== 测试完整擦除流程 ===")
    try:
        # 创建一个只包含测试帧的OCR结果子集
        test_ocr_result = {k: v for k, v in ocr_result.items() 
                          if int(k.split('/')[1][:4]) in test_frames}
        
        # 执行擦除
        remove_subtitles(
            ocr_result=test_ocr_result,
            fps=args.fps,
            total_frames=len(test_frames),
            config=config
        )
        print("擦除流程执行完成")
        
    except Exception as e:
        print(f"擦除过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n=== 调试完成 ===")
    print(f"请检查输出目录 {args.output_dir} 中的调试图像")
    print("1. *_1_ocr_box.png - 显示OCR检测框")
    print("2. *_2_mask.png - 显示生成的mask")
    print("3. *_2_masked.png - 显示mask在原图上的覆盖区域")
    print("4. *_3_resized.png 和 *_3_resized_mask.png - 显示STTN模型的输入")


