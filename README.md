# ClassySORT
    
YOLO v5(image segmentation) + vanilla SORT(multi-object tracker) implementation 
that is aware of the tracked object category.

This is for people who want a real-time multiple object tracker (MOT) 
that can track any kind of object with no additional training.

If you only need to track people, then I recommend YOLOv5 + DeepSORT implementations.
DeepSORT adds a separately trained neural network on top of SORT, 
which increases accuracy for human detections but decreases performance slightly.
