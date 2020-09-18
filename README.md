# Helmet.AI
Support system dedicated to monitoring the safety of workers on a construction site using Computer Vision.

## Summary of the results

ID | Train | Test | Head | Helmet | Mask | Headset | Chest | Vest | Person | mAP
-- | ----- | ---- | ---- | ------ | ---- | ------- | ----- | ---- | ------ | ---
Y3 | V | V  | 89.7% | 86.7% | 
Y4 | V | V  |
Y3 | R | R  |
Y4 | R | R  |
Y3 | V | R  |
Y4 | V | R  |
Y3 | VR | R |
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
