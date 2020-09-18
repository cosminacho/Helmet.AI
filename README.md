# Helmet.AI
Support system dedicated to monitoring the safety of workers on a construction site using Computer Vision.

## Objectives
- Personal Protective Equipment (PPE) detection (YOLOv4)
- object tracking
- face recognition


## PPE detection results

![Normal Detection - High Scores](https://github.com/cosminacho/Helmet.AI/blob/master/images/image1.png)
![Overlapped Objects](https://github.com/cosminacho/Helmet.AI/blob/master/images/image2.png)
![Weird Results](https://github.com/cosminacho/Helmet.AI/blob/master/images/image3.png)

## Summary of the (numerical) results

ID | Train | Test | Head | Helmet | Mask | Headset | Chest | Vest | Person | mAP
-- | ----- | ---- | ---- | ------ | ---- | ------- | ----- | ---- | ------ | ---
Y3 | V | V  | 89.7% | 86.7% | 75.5% | 89.0% | 89.7% | 90.0% | 89.7% | 87.2%
Y4 | V | V  | 
Y3 | R | R  | 44.1% | 52.2% | 42.3% | 62.0% | 59.1% | 60.7% | 80.6% | 57.3%
Y4 | R | R  |
Y3 | V | R  | 36.3% | 74.1% | 27.3% | 55.6% | 45.7% | 69.9% | 76.9% | 55.1%
Y4 | V | R  |
Y3 | VR | R | 78.8% | 73.3% | 66.3% | 74.0% | 74.7% | 78.6% | 87.1% | 76.1%
Y4 | VR | R |

## TODO / Work In Progress
- scientific article publication.
- hardware (NVidia Jetson Nano) & software (video streaming platform) integration on a real construction site from Romania.


## Referernces
- VW-PPE dataset + paper: http://aimir.isti.cnr.it/vw-ppe
- Colored helmets dataset: https://github.com/wujixiu/helmet-detection
- Darknet repository (training only): https://github.com/AlexeyAB/darknet
- Tensorflow implementation (testing): https://github.com/hunglc007/tensorflow-yolov4-tflite
- DeepSort implementation: https://github.com/nwojke/deep_sort
- Face recognition model: https://github.com/ageitgey/face_recognition
