# debug_erase.py
import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from modules.erase import debug_subtitle_erase, remove_subtitles

def test_frame_range(center_frame=485, frame_range=100):
    """
    测试指定中心帧附近的帧段
    
    参数:
    center_frame: int - 中心帧号（比如字幕出现的某一帧）
    frame_range: int - 向前和向后测试的帧数范围
    """
    # 计算测试范围
    start_frame = max(0, center_frame - frame_range)
    end_frame = center_frame + frame_range
    test_frames = list(range(start_frame, end_frame + 1))
    
    # 构建测试用的OCR结果
    test_ocr_result = {}
    # 在字幕实际出现的帧范围内添加OCR结果
    for i in range(473, 499):  # 原始字幕的帧范围
        # 注意这里的键值需要和实际文件路径分开处理
        frame_key = f"3/{i:04d}.png,0"  # OCR结果的键值
        frame_path = f"3/{i:04d}.png"    # 实际文件路径
        
        # 检查文件是否存在
        if not os.path.exists(frame_path):
            print(f"警告: 文件不存在 {frame_path}")
            continue
            
        test_ocr_result[frame_key] = {
            "box": [451, 1418, 623, 1518],
            "text": "什么"
        }
    
    # 配置信息
    config = {
        "erase": {
            "mask_expand": 30,
            "max_frame_length": 100,
            "min_frame_length": 2,
            "neighbor_stride": 5,  # 减小步长以获得更好的效果
            "ckpt_p": "./models/sttn.pth"
        }
    }
    
    # 创建输出目录
    output_dir = "debug_output_extended"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== 开始扩展范围测试 ===")
    print(f"测试帧范围: {start_frame} - {end_frame}")
    print(f"总帧数: {len(test_frames)}")
    print(f"字幕帧范围: 473 - 498")
    print(f"输出目录: {output_dir}")
    
    # 1. 保存原始帧作为参考
    print("\n1. 保存原始帧...")
    for frame_num in tqdm(test_frames):
        frame_path = f"3/{frame_num:04d}.png"
        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            if img is not None:
                cv2.imwrite(os.path.join(output_dir, f"original_{frame_num:04d}.png"), img)
    
    # 修改 extract_mask 函数的调用方式
    def process_frame_path(frame_path_with_suffix):
        """处理带后缀的帧路径"""
        return frame_path_with_suffix.split(',')[0]
    
    # 2. 执行调试流程
    print("\n2. 执行调试流程...")
    debug_subtitle_erase(
        test_frames=test_frames,
        ocr_result=test_ocr_result,
        fps=30.0,
        config=config,
        output_dir=os.path.join(output_dir, "debug")
    )
    
    # 3. 执行完整擦除流程
    print("\n3. 执行完整擦除流程...")
    try:
        # 创建一个处理后的OCR结果副本
        processed_ocr_result = {}
        for key, value in test_ocr_result.items():
            # 移除文件路径中的后缀
            new_key = process_frame_path(key)
            processed_ocr_result[new_key] = value
        
        remove_subtitles(
            ocr_result=processed_ocr_result,
            fps=30.0,
            total_frames=len(test_frames),
            config=config
        )
        print("擦除流程执行完成")
        
        # 4. 对比分析
        print("\n4. 进行对比分析...")
        for frame_num in tqdm(range(473, 499)):
            frame_path = f"3/{frame_num:04d}.png"
            if not os.path.exists(frame_path):
                print(f"跳过不存在的帧: {frame_path}")
                continue
                
            original = cv2.imread(os.path.join(output_dir, f"original_{frame_num:04d}.png"))
            processed = cv2.imread(frame_path)
            
            if original is not None and processed is not None:
                # 保存对比图
                comparison = np.hstack((original, processed))
                cv2.imwrite(
                    os.path.join(output_dir, f"comparison_{frame_num:04d}.png"),
                    comparison
                )
                
                # 计算差异
                diff = cv2.absdiff(original, processed)
                cv2.imwrite(
                    os.path.join(output_dir, f"diff_{frame_num:04d}.png"),
                    diff
                )
                
                # 计算修复区域的统计信息
                mask = np.zeros(original.shape[:2], dtype=np.uint8)
                cv2.rectangle(
                    mask,
                    (451, 1418),
                    (623, 1518),
                    255,
                    -1
                )
                mask_region = cv2.bitwise_and(diff, diff, mask=mask)
                mean_diff = cv2.mean(mask_region)[0]
                
                with open(os.path.join(output_dir, "analysis.txt"), "a") as f:
                    f.write(f"Frame {frame_num}: Mean difference in subtitle region: {mean_diff}\n")
        
    except Exception as e:
        print(f"处理过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")
    print(f"请检查输出目录 {output_dir} 中的结果：")
    print("1. original_*.png - 原始帧")
    print("2. debug/* - 调试过程的中间结果")
    print("3. comparison_*.png - 原始帧与处理后的对比")
    print("4. diff_*.png - 差异可视化")
    print("5. analysis.txt - 数值分析结果")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='扩展范围字幕擦除测试')
    parser.add_argument('--center', type=int, default=485, help='中心帧号')
    parser.add_argument('--range', type=int, default=100, help='向前和向后测试的帧数')
    args = parser.parse_args()
    
    test_frame_range(args.center, args.range)