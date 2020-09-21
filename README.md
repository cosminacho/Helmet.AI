# Helmet.AI
Support system dedicated to monitoring the safety of workers on a construction site using Computer Vision.

## Objectives
- Personal Protective Equipment (PPE) detection (YOLOv4)
- object tracking
- face recognition


## PPE detection

![Normal Detection - High Scores](https://github.com/cosminacho/Helmet.AI/blob/master/images/image1.png)
![Overlapped Objects](https://github.com/cosminacho/Helmet.AI/blob/master/images/image2.png)
![Weird Results](https://github.com/cosminacho/Helmet.AI/blob/master/images/image3.png)

More image and video results [here](https://drive.google.com/drive/folders/12nFQN60InowsPQm1O2cIBApnOicgxeLK?usp=sharing)

## Summary of the (numerical) results

ID | Train | Test | Head | Helmet | Mask | Headset | Chest | Vest | Person | mAP
-- | ----- | ---- | ---- | ------ | ---- | ------- | ----- | ---- | ------ | ---
Y3A | V | V  | 89.7% | 86.7% | 75.5% | 89.0% | 89.7% | 90.0% | 89.7% | 87.2%
Y4M | V | V  | 96.6% | 97.1% | 86.1% | 98.7% | 96.5% | 98.0% | 95.0% | 95.5%
Y3A | R | R  | 44.1% | 52.2% | 42.3% | 62.0% | 59.1% | 60.7% | 80.6% | 57.3%
Y4M | R | R  | ? | ? | ? | ? | ? | ? | ? | ? 
Y3A | V | R  | 36.3% | 74.1% | 27.3% | 55.6% | 45.7% | 69.9% | 76.9% | 55.1%
Y4M | V | R  | 42.5% | 58.2% | 16.1% | 69.9% | 36.8% | 60.3% | 74.6% | 51.2%
Y3A | VR | R | 78.8% | 73.3% | 66.3% | 74.0% | 74.7% | 78.6% | 87.1% | 76.1%
Y4M | VR | R | 79.4% | 83.9% | 75.6% | 91.9% | 82.9% | 85.2% | 87.1% | 83.7%

### Observations:
- V - Virtual dataset, R - Real-world dataset, VR - trained on V and fine-tuned on R
- Y3A - the YOLOv3 model of the authors; Y4M - my YOLOv4 model.
- "the authors" are the ones who created the virtual dataset in the game GTA-V.

## Weights: [here](https://drive.google.com/drive/folders/1g_0nnMd6LdpWCTJqXl3uDUBXgWrUdLlB?usp=sharing)

## Datasets: [here](https://drive.google.com/drive/folders/1jE9HQJ2Zd7xK5N3gyoleTdzdzQ5n3XcQ?usp=sharing)


## TODO / Work In Progress
- scientific article publication.
- hardware (NVidia Jetson Nano) & software (video streaming platform) integration on a real construction site from Romania. ([EFdeN](https://efden.org/))


## Referernces
- VW-PPE dataset + paper: http://aimir.isti.cnr.it/vw-ppe
- Colored helmets dataset: https://github.com/wujixiu/helmet-detection
- Darknet repository (training only): https://github.com/AlexeyAB/darknet
- YOLOv4 Tensorflow implementation (testing): https://github.com/hunglc007/tensorflow-yolov4-tflite
- YOLOv3 Tensorflow implementation: https://github.com/zzh8829/yolov3-tf2
- DeepSort implementation: https://github.com/nwojke/deep_sort
- Face recognition model: https://github.com/ageitgey/face_recognition
