# ClassySORT

## Introduction
ClassySORT is a real-time multi-object tracker that can track any kind of object without additional training.
It implements [YOLOv5](https://github.com/ultralytics/yolov5/wiki) and [SORT](https://github.com/abewley/sort), with no modifications to YOLOv5 and minor modifications to SORT.

If 1) you only need to track people, or 2)you have the resources to train from scratch,
then I recommend [YOLOv5 + DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).
DeepSORT adds a separately trained neural network on top of SORT, which increases accuracy for human detections but slightly decreases performance.

For a NVIDIA Jetson-optimized version of YOLOv5 + DeepSORT, see [FastMOT](https://github.com/GeekAlexis/FastMOT)
