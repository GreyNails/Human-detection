# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import sys
import os


# export PYTHONPATH="/home/usr/dell/DataTool-HumanCentric/detection_detectron2:$PYTHONPATH"
# sys.path.insert(0, os.path.abspath(".."))
# from vision.fair.detectron2.demo.predictor import VisualizationDemo

# os.chdir("..")


# detectron2_path = os.path.abspath("/home/usr/dell/DataTool-HumanCentric/detection_detectron2")
# sys.path.append(detectron2_path)
# os.chdir(detectron2_path)

from detectron2.demo.predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        # default="/home/usr/dell/DataTool-HumanCentric/detection_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        # default="/home/usr/dell/DataTool-HumanCentric/detection_detectron2/configs/LVISv1-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml",
        default="/home/usr/dell/DataTool-HumanCentric/detection_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")


    parser.add_argument(
        "--input",
        nargs="+",
        # default="/home/dell/Human_centric/test_IMG/gt/*.png",
        # default="/home/dell/Human_centric/detectron2/demo/test_img/1.png",
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
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        # default=[],
        # default="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl ",
        # default=["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],

        default=["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
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


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    human_img_list=[]

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            
            try:
                img = read_image(path, format="BGR")
            except:
                print (f"Error processing {path}")
                continue
            # img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output,human_exit = demo.run_on_image(img)
            if human_exit==True:
                human_img_list.append(path)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    (
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished"
                    ),
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            # else:
            #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            #     if cv2.waitKey(0) == 27:
            #         break  # esc to quit
        print(human_img_list)
        print(len(human_img_list))
        file_path = "/home/dell/Human_centric/list_cgl.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(human_img_list, f, ensure_ascii=False, indent=4)
        
        
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        
        # for vis in tqdm.tqdm(demo.run_on_video(cam)):
            # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # cv2.imshow(WINDOW_NAME, vis)
            # if cv2.waitKey(1) == 27:
            #     break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
        
        
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        # for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        #     if args.output:
        #         output_file.write(vis_frame)
        #     else:
        #         cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
        #         cv2.imshow(basename, vis_frame)
        #         if cv2.waitKey(1) == 27:
        #             break  # esc to quit
        # video.release()
        # if args.output:
        #     output_file.release()
        # else:
        #     cv2.destroyAllWindows()
            
            
                # 假设 args.output 为 None 或 False 时保存图片
        output_dir = "output_images"  # 指定保存图像的目录
        os.makedirs(output_dir, exist_ok=True)  # 创建保存图像的文件夹

        for i, vis_frame in tqdm.tqdm(enumerate(demo.run_on_video(video)), total=num_frames):
            # 生成保存文件的名称，例如 frame_001.png, frame_002.png
            frame_filename = os.path.join(output_dir, f"frame_{i+1:03d}.png")

            # 保存图像到指定的文件夹
            cv2.imwrite(frame_filename, vis_frame)

        # 完成后释放视频资源
        video.release()


def main_2() -> None:
    mp.set_start_method("spawn", force=True)
    
    args = get_parser().parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    human_img_list=[]

    if args.input:
        
        input_path=args.input
        input_path=input_path[0]
        image_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.lower().endswith('.png')]
            
        for path in tqdm.tqdm(image_files, disable=not args.output):
            # use PIL, to be consistent with evaluation
            
            try:
                img = read_image(path, format="BGR")
            except:
                print (f"Error processing {path}")
                continue
            # img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output,human_exit,human_box,image_height,image_width = demo.run_on_image(img)
            if human_exit==True:
                # num_human_boxes = len(human_box)
                # box=human_box.tolist()
                # result = {
                # "num_human_boxes": num_human_boxes,
                # "path": path,
                # "human_box": human_box,
                # "image_height": image_height,
                # "image_width": image_width
                # }
                human_img_list.append(path)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    (
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished"
                    ),
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            # else:
            #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            #     if cv2.waitKey(0) == 27:
            #         break  # esc to quit
        # print(human_img_list)
        print(len(human_img_list))
        # file_path = "/storage/human_psd/img_with_human/fp_v1_with_human.json"
        file_path = args.json

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(human_img_list, f, ensure_ascii=False, indent=4)
        

        


if __name__ == "__main__":
    main_2()  # pragma: no cover



#  python demo.py --input /storage/human_psd/img/test --output /storage/human_psd/img_with_human/test --json /storage/human_psd/img_with_human/test.json
