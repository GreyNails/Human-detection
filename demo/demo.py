# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
from multiprocessing import Pool, Manager
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
from functools import partial
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import sys
import os

from detectron2.demo.predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

# export PYTHONPATH="/home/usr/dell/DataTool-HumanCentric/detection_detectron2:$PYTHONPATH"
def setup_cfg(args):
    """设置配置"""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/usr/dell/DataTool-HumanCentric/detection_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default="/storage/human_psd/img/fp_v2/*.png",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/storage/human_psd/img_with_human/fp_v2",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--json",
        default="/storage/human_psd/img_with_human/fp_v2.json",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker processes for multiprocessing",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


def init_worker(config_file, opts, confidence_threshold):
    """初始化worker进程"""
    global demo, logger
    
    # 设置logger
    setup_logger(name="fvcore")
    logger = setup_logger()
    
    # 创建args对象用于setup_cfg
    class Args:
        pass
    
    args = Args()
    args.config_file = config_file
    args.opts = opts
    args.confidence_threshold = confidence_threshold
    
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)


# def process_single_image(image_info):
#     """处理单张图片"""
#     path, output_dir = image_info
    
#     try:
#         img = read_image(path, format="BGR")
#     except Exception as e:
#         print(f"Error processing {path}: {e}")
#         return None
    
#     start_time = time.time()
#     predictions, visualized_output, human_exit, human_box, image_height, image_width = demo.run_on_image(img)
    
#     processing_time = time.time() - start_time
    
#     # 保存输出图像
#     if output_dir:
#         if os.path.isdir(output_dir):
#             out_filename = os.path.join(output_dir, os.path.basename(path))
#             visualized_output.save(out_filename)
    
#     # 记录处理信息
#     num_instances = len(predictions["instances"]) if "instances" in predictions else 0
#     logger.info(f"{path}: detected {num_instances} instances in {processing_time:.2f}s")
    
#     # 返回结果
#     if human_exit:
#         return {
#             'path': path,
#             'has_human': True,
#             'processing_time': processing_time,
#             'num_instances': num_instances
#         }
#     else:
#         return {
#             'path': path,
#             'has_human': False,
#             'processing_time': processing_time,
#             'num_instances': num_instances
#         }
        

def process_single_image(image_info):
    """处理单张图片"""
    path, output_dir = image_info
    
    try:
        img = read_image(path, format="BGR")
        
        # 添加图像尺寸验证
        if img is None or img.size == 0:
            logger.warning(f"图像为空或无法读取: {path}")
            return None
            
        # 检查图像尺寸
        height, width = img.shape[:2]
        if height <= 0 or width <= 0:
            logger.warning(f"图像尺寸无效 (h={height}, w={width}): {path}")
            return None
            
        # 可选：设置最小尺寸限制
        min_size = 10  # 最小10x10像素
        if height < min_size or width < min_size:
            logger.warning(f"图像尺寸过小 (h={height}, w={width}): {path}")
            return None
            
    except Exception as e:
        logger.error(f"读取图像失败 {path}: {e}")
        return None
    
    try:
        start_time = time.time()
        predictions, visualized_output, human_exit, human_box, image_height, image_width = demo.run_on_image(img)
        processing_time = time.time() - start_time
        
        # 保存输出图像
        if output_dir and visualized_output:
            if os.path.isdir(output_dir):
                out_filename = os.path.join(output_dir, os.path.basename(path))
                visualized_output.save(out_filename)
        
        # 记录处理信息
        num_instances = len(predictions["instances"]) if "instances" in predictions else 0
        logger.info(f"{path}: 检测到 {num_instances} 个实例，耗时 {processing_time:.2f}秒")
        
        # 返回结果
        return {
            'path': path,
            'has_human': human_exit,
            'processing_time': processing_time,
            'num_instances': num_instances
        }
        
    except Exception as e:
        logger.error(f"处理图像时出错 {path}: {e}")
        return None

def process_images_batch(image_paths, args, num_workers=4):
    """批量处理图像（多进程版本）"""
    # 准备输入数据
    image_infos = [(path, args.output) for path in image_paths]
    
    # 创建进程池
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.config_file, args.opts, args.confidence_threshold)
    ) as pool:
        # 使用tqdm显示进度
        results = list(tqdm.tqdm(
            pool.imap(process_single_image, image_infos),
            total=len(image_infos),
            desc="Processing images"
        ))
    
    # 过滤出包含人体的图像
    human_img_list = [
        result['path'] for result in results 
        if result is not None and result['has_human']
    ]
    
    # 统计信息
    total_processed = len([r for r in results if r is not None])
    total_with_human = len(human_img_list)
    avg_processing_time = np.mean([r['processing_time'] for r in results if r is not None])
    
    print(f"\nProcessing Summary:")
    print(f"Total images processed: {total_processed}")
    print(f"Images with humans: {total_with_human}")
    print(f"Average processing time: {avg_processing_time:.2f}s")
    
    return human_img_list



def main() -> None:
    """主函数"""
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    
    # 设置主进程的logger
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    
    # 创建输出目录
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
    
    human_img_list = []
    
    if args.input:
        # 获取所有输入图像路径
        input_path = args.input[0]
        if os.path.isdir(input_path):
            # 如果是目录，获取所有png文件
            image_files = [
                os.path.join(input_path, file) 
                for file in os.listdir(input_path) 
                if file.lower().endswith('.png')
            ]
        else:
            # 如果是glob模式
            image_files = glob.glob(os.path.expanduser(input_path))
        
        if not image_files:
            logger.error("No input images found!")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        logger.info(f"Using {args.num_workers} worker processes")
        
        # 多进程处理图像
        human_img_list = process_images_batch(
            image_files, 
            args, 
            num_workers=args.num_workers
        )
        
        # 保存结果到JSON文件
        logger.info(f"Saving results to {args.json}")
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(human_img_list, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Found {len(human_img_list)} images with humans")
        
    elif args.webcam:
        logger.error("Webcam mode not supported in multiprocessing version")
        
    elif args.video_input:
        logger.error("Video input not supported in multiprocessing version")


def test_opencv_video_format(codec, file_ext):
    """测试OpenCV视频格式"""
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    main()  # pragma: no cover


# 使用示例：
# python demo_multiprocess.py --input /storage/human_psd/img/test --output /storage/human_psd/img_with_human/test --json /storage/human_psd/img_with_human/test.json --num-workers 8