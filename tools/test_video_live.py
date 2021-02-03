#conda environment: conda_mmdetection
#command line to run: python3 tools/test_video_live.py configs/masktrack_rcnn_r50_fpn_1x_live.py models/MaskTrackRCNN_epoch_12.pth --out outputs/results.pkl
#to update after changes to code ---> !pip install .

import argparse
import torch
import mmcv
from mmcv.runner import load_checkpoint, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json_videoseg, ytvos_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

import pyrealsense2 as rs
import numpy as np
import cv2
from torch.utils.data import DataLoader
from functools import partial
import os
import shutil
import json
from mmdet.apis import show_result
import time


def single_test(model, data_loader, show=False, save_path=''):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result_demo(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES,
                                     save_vis = True,
                                     save_path = save_path,
                                     is_video = True)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_frame(model, data_loader, show=False, save_path='', save_name_alt=None, color_image=None):
    time_before_frame_1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    model.eval()
    results = []
    dataset = data_loader.dataset
    # this operation takes 140ms?
    data = list(data_loader)[-1]

    time_before_frame_2 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    with torch.no_grad():
        result = model(return_loss=False, rescale=not show, **data)
    results.append(result)

    time_before_frame_3 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    blended_img = None
    if show:
        img = model.module.show_result_demo(data, result, dataset.img_norm_cfg,
                                 dataset=dataset.CLASSES,
                                 save_vis = True,
                                 save_path = save_path,
                                 is_video = True,
                                 save_path_alt=True,
                                 save_name_alt=save_name_alt,
                                 color_image=color_image)

    time_after_frame = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    total_frame_time = time_after_frame - time_before_frame_1
    total_frame_time_delta = time_after_frame - time_before_frame_2
    total_frame_time_forward = time_before_frame_3 - time_before_frame_2
    #print("total frame time 1 : ", total_frame_time/1000000.0)
    print("frame forward time : ", total_frame_time_forward/1000000.0)

    return results, img



def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save_path',
        type=str,
        help='path to save visual result')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--load_result',
        action='store_true',
        help='whether to load existing result')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def build_dataloader_live(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedGroupSampler(dataset, imgs_per_gpu, world_size,
                                          rank)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if not kwargs.get('shuffle', True):
            sampler = None
        else:
            sampler = GroupSampler(dataset, imgs_per_gpu)
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader

def show_result_blended(model, img, result, score_thr=0.3):
    if hasattr(model, 'module'):
        model = model.module
    #img = model.show_result(img, result, score_thr=score_thr)
    img = model.module.show_result_demo(data, result, dataset.img_norm_cfg,
                             dataset=dataset.CLASSES,
                             save_vis = True,
                             save_path = save_path,
                             is_video = True,
                             save_path_alt=True,
                             save_name_alt=save_name_alt)
    blended_img = mmcv.bgr2rgb(img)
    return blended_img


def main():
    args = parse_args()
    data_root = '/home/bryanbed/Projects/RoMeLa_Vision/MaskTrackRCNN/data/'
    ann_root = '/home/bryanbed/Projects/RoMeLa_Vision/MaskTrackRCNN/data/annotations/instances_live_sub.json'
    config_root = '/home/bryanbed/Projects/RoMeLa_Vision/MaskTrackRCNN/configs/masktrack_rcnn_r50_fpn_1x_live.py'

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # framerate = 6
    # reswidth = 1280
    # resheight = 720
    framerate = 30
    reswidth = 640
    resheight = 480
    cfg.data.test.img_scale=(reswidth,resheight)

    #dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    assert args.gpus == 1
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])

    # configure video pipeline
    pipe = rs.pipeline()
    config = rs.config()


    config.enable_stream(rs.stream.depth, reswidth, resheight, rs.format.z16, framerate)
    config.enable_stream(rs.stream.color, reswidth, resheight, rs.format.bgr8, framerate)
    profile = pipe.start(config)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(6):
        pipe.wait_for_frames()

    loopID = 0
    run = True
    color_frame_path = data_root + 'live/raw_color/'
    if os.path.isdir(color_frame_path):
        shutil.rmtree(color_frame_path)
    os.makedirs(color_frame_path)
    print(color_frame_path)
    depth_frame_path = data_root + 'live/raw_depth/'
    if os.path.isdir(depth_frame_path):
        shutil.rmtree(depth_frame_path)
    os.makedirs(depth_frame_path)
    print(depth_frame_path)
    save_path = data_root + 'live/processed/'
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print(save_path)

    with open(ann_root) as f:
        annotations = json.load(f)
    annotations["videos"][0]["width"] = reswidth
    annotations["videos"][0]["height"] = resheight
    os.remove(ann_root)
    json_object = json.dumps(annotations, indent=4)
    with open(ann_root, 'w') as f:
         f.write(json_object)

    start_time_prime = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    frame_list = []

    # print("Time 1: ", (time.clock_gettime_ns(time.CLOCK_MONOTONIC)-start_time_prime)/1000000.0)

    try:
        while run == True:
            #print("Time 1: ", (time.clock_gettime_ns(time.CLOCK_MONOTONIC)-start_time_prime)/1000000.0)
            loopID += 1
            if loopID > 50:
                run = False
            frameID_str = 'frame_' + str(loopID) +'.jpg'
            frames = pipe.wait_for_frames()
            #for f in frames:
            #    total_frames += 1

            #print("Time 2: ", (time.clock_gettime_ns(time.CLOCK_MONOTONIC)-start_time_prime)/1000000.0)
            # preprocess color image stream
            color_frame = frames.get_color_frame().get_data()
            color_image = np.asanyarray(color_frame)
            frame_name = color_frame_path + frameID_str
            cv2.imwrite(frame_name, color_image)

            # align depth frame to color frame
            align = rs.align(rs.stream.color)
            frameset = align.process(frames)
            aligned_depth_frame = frameset.get_depth_frame()

            # preprocess colorized depth image stream
            colorizer = rs.colorizer(color_scheme=5)
            depth_frame_colored = colorizer.colorize(aligned_depth_frame)
            depth_image_colored = np.asanyarray(depth_frame_colored.get_data())
            frame_name = depth_frame_path + frameID_str
            cv2.imwrite(frame_name, depth_image_colored)

            #print(ann_root)
            #json_frame_name = 'raw_color/frame_' + str(loopID) + 'jpeg'
            with open(ann_root) as f:
                annotations = json.load(f)
            #annotations = json.load(ann_root)
            #print(annotations["videos"][0]["file_names"][0])
            # json_frame_name = "raw_color/frame_" + str(loopID) + ".jpg"
            #print(annotations["videos"][0]["file_names"][0])
            next_frame = "raw_color/" + frameID_str
            frame_list.append(next_frame)
            annotations["videos"][0]["file_names"] = frame_list
            annotations["videos"][0]["width"] = reswidth
            annotations["videos"][0]["height"] = resheight
            #print(annotations)
            #print(annotations["videos"][0]["file_names"][0])

            os.remove(ann_root)
            json_object = json.dumps(annotations, indent=4)
            with open(ann_root, 'w') as f:
                 f.write(json_object)

            #print("annotation:", annotations["videos"][0]["file_names"][0])

            #print("Time 3: ", (time.clock_gettime_ns(time.CLOCK_MONOTONIC)-start_time_prime)/1000000.0)
            # make a dataloader with a single image
            dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
            data_loader = build_dataloader_live(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                num_gpus=1,
                dist=False,
                shuffle=False)

            #print("Time 4: ", (time.clock_gettime_ns(time.CLOCK_MONOTONIC)-start_time_prime)/1000000.0)
            result, blended_img = single_frame(model, data_loader, show=True, save_path=save_path, save_name_alt=frameID_str, color_image=color_image)
            #blended_img = show_result_blended(model, color_image, result[0], score_thr=0.3)

            # stack and display images
            #images = np.hstack((color_image, depth_image_colored))
            #images = np.hstack((color_image, img))
            cv2.namedWindow('D415 Example 1', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('D415 Example 1', blended_img)
            #cv2.namedWindow('D415 Example 2', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('D415 Example 2', blended_img)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            #print("Time 5: ", (time.clock_gettime_ns(time.CLOCK_MONOTONIC)-start_time_prime)/1000000.0)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        end_time_prime = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        print("total run time: ", (end_time_prime-start_time_prime)/1000000000.0)
        print("total frames: ", loopID)
        print("average fps: ", loopID / ((end_time_prime-start_time_prime)/1000000000.0))
        pipe.stop()

    # data_loader = build_dataloader(
    #     dataset,
    #     imgs_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     num_gpus=1,
    #     dist=False,
    #     shuffle=False)


    # if args.out:
    #     if not args.load_result:
    #         print('writing results to {}'.format(args.out))
    #         mmcv.dump(outputs, args.out)
    #     eval_types = args.eval
        # if eval_types:
        #     print('Starting evaluate {}'.format(' and '.join(eval_types)))
        #     if not isinstance(outputs[0], dict):
        #         result_file = args.out + '.json'
        #         results2json_videoseg(dataset, outputs, result_file)
        #         ytvos_eval(result_file, eval_types, dataset.ytvos)
        #     else:
        #         NotImplemented


if __name__ == '__main__':
    main()
