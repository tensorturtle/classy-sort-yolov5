# ℂ𝕝𝕒𝕤𝕤𝕪𝕊𝕆ℝ𝕋
ClassySORT is a real-time multi-object tracker (MOT) that works for any kind of object (not just people).

ClassySORT is designed to be a simple MOT to use for your own projects. And bcause the YOLO detector pretrained on COCO, ClassySORT can detect and track 80 different kinds of common objects 'out of the box'. No supercomputer needed to use ClassySORT.

Modifying it is exactly the same process as training YOLO with your own dataset.

by Jason Sohn

## Introduction
ClassySORT implements 
+[ultralytics/YOLOv5](https://github.com/ultralytics/yolov5/wiki) with no modifications
+[abewley/SORT](https://github.com/abewley/sort) with minor modifications

If you only need to track people, or have the resources to train a model from scratch with your own dataset, see 'More Complex MOTs' section below.

## Using ClassySORT

### Install Requirements
Python 3.8 or later with all requirements.txt. To install un:
`pip install -r requirements.txt`

### Run Tracking

NOTE: The saved results.txt is not MOT compliant.


## Implementation Details

### Modifications to SORT

The original implementation of SORT threw away YOLO's object class information (0: person, 1: bike, etc.).
I wanted to keep that information, so I added a `detclass` attribute to `KalmanBoxTracker` object in `sort.py`.
With this modification, SORT returns tracked detections in the format:
`[x_left_top, y_left_top, x_right_bottom, y_right_bottom, object_category, object_identification]`



## More Complex MOTs
If you only need to track people, or have the resources to train a model from scratch with your own dataset, then I recommend [bostom/Yolov5_DeepSort_PyTorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).
DeepSORT adds a separately trained neural network on top of SORT, which increases accuracy for human detections but slightly decreases performance.
It also means that using your custom dataset involves training both YOLO and DeepSORT's 'deep association metric'

For a 'bag of tricks' optimized version of YOLOv5 + DeepSORT, see [GeekAlexis/FastMOT](https://github.com/GeekAlexis/FastMOT)

## License

ClassySORT is released under the GPL License to promote the open use of the tracker and future improvements.

## Visual Identity
ClassySORT theme color: #7FFFD4
