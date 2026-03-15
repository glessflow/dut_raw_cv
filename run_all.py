import sys
from isp_pipeline import ISP_Pipeline
from path import Path
import os
import time
import numpy as np
import cv2
import glob
import multiprocessing
from functools import partial

sys.path.insert(0, os.path.dirname(__file__) + '/algorithm')
sys.path.insert(0, os.path.dirname(__file__) + '/config')
sys.path.insert(0, os.path.dirname(__file__) + '/assets')
sys.path.insert(0, os.path.dirname(__file__) + '/test_images')
sys.path.insert(0, os.path.dirname(__file__))
BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24

def load_raw(name):
    """
    加载并预处理RAW图片
    """
    assert os.path.exists(name) and name.endswith('.raw'), f'Invalid raw name < {name} >!'
    
    # 读取RAW文件
    raw = np.fromfile(name, dtype=np.uint8)
    print(f"RAW文件读取完成，数据大小: {raw.size}")
    
    # 重塑数组形状为三维（高度 x 宽度 x 3）
    expected_size = 1856 * 2880 * 3
    if raw.size != expected_size:
        raise ValueError(f"RAW数据大小不匹配，预期 {expected_size}，实际 {raw.size}")
    
    raw = raw.reshape(1856, 2880, 3).astype(np.float32)
    print(f"RAW数据重塑完成，形状: {raw.shape}")
    
    # 合并通道为24位整数
    raw = np.split(raw, 3, axis=2)
    raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16)
    raw = raw.squeeze()  # 去除多余的维度，形状变为 (1856, 2880)
    print(f"通道合并完成，形状: {raw.shape}")
    
    # 归一化到 [0, 1]
    raw = raw / (BIT24 - 1)
    print(f"数据归一化完成，范围: [0, 1]")
    
    # 转换为16位无符号整数
    raw = (raw * 65535).astype(np.uint16)
    print(f"数据类型转换完成，新类型: {raw.dtype}")
    
    return raw

def process_single_image(raw_file, isp, output_folder):
    """
    单张图片的处理函数
    """
    # 获取文件名（不带扩展名）
    file_name = os.path.splitext(os.path.basename(raw_file))[0]
    output_path = os.path.join(output_folder, f"{file_name}.jpg")
    
    # 加载并预处理RAW图片
    raw = load_raw(raw_file)
    print(f"图片加载并预处理完成，形状: {raw.shape}, 类型: {raw.dtype}")
    
    # 运行ISP Pipeline并传递输入数据，不保存中间结果
    print(f"开始运行ISP Pipeline for {raw_file}...")
    output = isp.run(input_data=raw, save_intermediate=False)
    print(f"ISP Pipeline运行完成，输出形状: {output.shape}")
    
    # 保存处理后的图片
    cv2.imwrite(output_path, output)
    print(f"处理完成，结果保存在: {output_path}")

def run_batch_processing_parallel(input_folder: str, output_folder: str, num_processes: int = None):
    """
    使用多进程并行处理文件夹中的所有 .raw 文件
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        num_processes: 进程数量，默认为 CPU 核心数
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取输入文件夹中的所有 .raw 文件
    raw_files = glob.glob(os.path.join(input_folder, '*.raw'))
    
    if not raw_files:
        print(f"未找到任何 .raw 文件，请检查输入文件夹: {input_folder}")
        return
    
    # 配置文件路径
    root_path = Path(os.path.abspath(__file__)).parent
    yaml_path = root_path / 'config' / 'isp_config.yaml'
    
    # 初始化 ISP Pipeline
    isp = ISP_Pipeline(config_path=yaml_path)
    
    # 使用多进程并行处理
    num_processes = num_processes or multiprocessing.cpu_count()
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 将任务分配到多个进程中
        pool.map(partial(process_single_image, isp=isp, output_folder=output_folder), raw_files)

if __name__ == "__main__":
    # 示例：处理输入文件夹中的所有 .raw 文件，并保存到输出文件夹
    input_folder = r"F:\data"  # 替换为你的输入文件夹路径
    output_folder = r"F:\data_good\yolodata\images\train"    # 替换为你的输出文件夹路径
    
    # 运行并行处理
    run_batch_processing_parallel(input_folder, output_folder,num_processes=8)