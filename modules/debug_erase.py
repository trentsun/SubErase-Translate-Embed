# debug_erase.py
import os
import sys
import json
import argparse
from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from modules.erase import remove_subtitles

def process_frames(center_frame=485, frame_range=100):
    """
    处理指定中心帧附近的帧段
    
    参数:
    center_frame: int - 中心帧号
    frame_range: int - 向前和向后处理的帧数范围
    """
    # 计算处理范围
    start_frame = max(0, center_frame - frame_range)
    end_frame = center_frame + frame_range
    test_frames = list(range(start_frame, end_frame + 1))
    
    # 构建OCR结果
    test_ocr_result = {}
    for i in range(473, 499):
        frame_path = f"3/{i:04d}.png"
        if not os.path.exists(frame_path):
            print(f"警告: 文件不存在 {frame_path}")
            continue
            
        test_ocr_result[frame_path] = {
            "box": [451, 1418, 623, 1518],
            "text": "什么"
        }
    
    # 配置信息
    config = {
        "erase": {
            "mask_expand": 30,
            "max_frame_length": 100,
            "min_frame_length": 2,
            "neighbor_stride": 5,
            "ckpt_p": "./models/sttn.pth"
        }
    }
    
    print(f"\n=== 开始处理 ===")
    print(f"处理帧范围: {start_frame} - {end_frame}")
    print(f"总帧数: {len(test_frames)}")
    print(f"字幕帧范围: 473 - 498")
    
    try:
        remove_subtitles(
            ocr_result=test_ocr_result,
            fps=30.0,
            total_frames=len(test_frames),
            config=config
        )
        print("处理完成")
        
    except Exception as e:
        print(f"处理过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 处理完成 ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='字幕擦除处理')
    parser.add_argument('--center', type=int, default=485, help='中心帧号')
    parser.add_argument('--range', type=int, default=100, help='向前和向后处理的帧数')
    args = parser.parse_args()
    
    process_frames(args.center, args.range)