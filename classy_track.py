"""
    ClassySORT
    
    YOLO v5(image segmentation) + vanilla SORT(multi-object tracker) implementation 
    that is aware of the tracked object category.
    
    This is for people who want a real-time multiple object tracker (MOT) 
    that can track any kind of object with no additional training.
    
    If you only need to track people, then I recommend YOLOv5 + DeepSORT implementations.
    DeepSORT adds a separately trained neural network on top of SORT, 
    which increases accuracy for human detections but decreases performance slightly.
    
    
    Copyright (C) 2020-2021 Jason Sohn tensorturtle@gmail.com
    
    
    === start GNU License ===
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    === end GNU License ===
"""

# python interpreter searchs these subdirectories for modules
import sys
sys.path.insert(0, './yolov5')
sys.path.insert(0, './sort')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

#yolov5
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized

#SORT
import skimage
from sort import *

torch.set_printoptions(precision=3)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        cat = int(categories[i]) if categories is not None else 0
        
        id = int(identities[i]) if identities is not None else 0
        
        color = compute_color_for_labels(id)
        
        label = f'{names[cat]} | {id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def detect(opt, *args):
    out, source, weights, view_img, save_txt, imgsz, save_img, sort_max_age, sort_min_hits, sort_iou_thresh= \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_img, opt.sort_max_age, opt.sort_min_hits, opt.sort_iou_thresh
    
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    # Initialize SORT
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) # {plug into parser}
    
    
    # Directory and CUDA settings for yolov5
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolov5 model
    model = torch.load(weights, map_location=device)['model'].float() #load to FP32. yolov5s.pt file is a dictionary, so we retrieve the model by indexing its key
    model.to(device).eval()
    if half:
        model.half() #to FP16

    # Set DataLoader
    vid_path, vid_writer = None, None
    
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)
    
    # get names of object categories from yolov5.pt model
    names = model.module.names if hasattr(model, 'module') else model.names 
    
    # Run inference
    t0 = time.time()
    img = torch.zeros((1,3,imgsz,imgsz), device=device) #init img
    
    # Run once (throwaway)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    
    save_path = str(Path(out))
    txt_path = str(Path(out))+'/results.txt'
    
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset): #for every frame
        img= torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() #unint8 to fp16 or fp32
        img /= 255.0 #normalize to between 0 and 1.
        if img.ndimension()==3:
            img = img.unsqueeze(0)
            
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0] 

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        
        # Process detections
        for i, det in enumerate(pred): #for each detection in this frame
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            
            s += f'{img.shape[2:]}' #print image size and detection report
            save_path = str(Path(out) / Path(p).name)

            # Rescale boxes from img_size (temporarily downscaled size) to im0 (native) size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            
            for c in det[:, -1].unique(): #for each unique object category
                n = (det[:, -1] ==c).sum() #number of detections per class
                s += f' - {n} {names[int(c)]}'

            dets_to_sort = np.empty((0,6))

            # Pass detections to SORT
            # NOTE: We send in detected object class too
            for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
            print('\n')
            print('Input into SORT:\n',dets_to_sort,'\n')

            # Run SORT
            tracked_dets = sort_tracker.update(dets_to_sort)

            print('Output from SORT:\n',tracked_dets,'\n')

            
            # draw boxes for visualization
            if len(tracked_dets)>0:
                bbox_xyxy = tracked_dets[:,:4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                draw_boxes(im0, bbox_xyxy, identities, categories, names)
                
            # Write detections to file. NOTE: Not MOT-compliant format.
            if save_txt and len(tracked_dets) != 0:
                for j, tracked_dets in enumerate(tracked_dets):
                    bbox_x1 = tracked_dets[0]
                    bbox_y1 = tracked_dets[1]
                    bbox_x2 = tracked_dets[2]
                    bbox_y2 = tracked_dets[3]
                    category = tracked_dets[4]
                    u_overdot = tracked_dets[5]
                    v_overdot = tracked_dets[6]
                    s_overdot = tracked_dets[7]
                    identity = tracked_dets[8]
                    
                    with open(txt_path, 'a') as f:
                        f.write(f'{frame_idx},{bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2},{category},{u_overdot},{v_overdot},{s_overdot},{identity}\n')
                
            print(f'{s} Done. ({t2-t1})')    
            # Stream image results(opencv)
            if view_img:
                cv2.imshow(p,im0)
                if cv2.waitKey(1)==ord('q'): #q to quit
                    raise StopIteration
            # Save video results
            if save_img:
                print('saving img!')
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1080,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-img', action='store_true',
                        help='save video file to output folder (disable for speed)')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[i for i in range(80)], help='filter by class') #80 classes in COCO dataset
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    
    #SORT params
    parser.add_argument('--sort-max-age', type=int, default=5,
                        help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=2,
                        help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.2,
                        help='intersection-over-union threshold between two frames for association')
    
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
