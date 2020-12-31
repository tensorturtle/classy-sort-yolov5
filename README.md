# ClassySORT

## Introduction
ClassySORT is a real-time multi-object tracker that can track any kind of object without additional training.
It implements [ultralytics/YOLOv5](https://github.com/ultralytics/yolov5/wiki) and [abewleySORT](https://github.com/abewley/sort), with no modifications to YOLOv5 and minor modifications to SORT.

If you only need to track people, or have the resources to train a model from scratch with your own dataset,
then I recommend [bostom/Yolov5_DeepSort_PyTorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).
DeepSORT adds a separately trained neural network on top of SORT, which increases accuracy for human detections but slightly decreases performance.

For a NVIDIA Jetson-optimized version of YOLOv5 + DeepSORT, see [GeekAlexis/FastMOT](https://github.com/GeekAlexis/FastMOT)
