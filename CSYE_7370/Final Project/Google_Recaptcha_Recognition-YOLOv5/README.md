# Google Recaptcha Recognition using YOLOv5

## Introduction

### Team Member
Yiyun Chen 
NUID:001557532

### About Capcha Recognition
It can be really hard to specify the label of capcha pictures: “Select all images with a bus/stop signs/a bicycle”, and sometimes we human may be judged as AI as we failed to choose all images correctly. I wonder how good a real AI would perform at this job, so I planned to use deep learning algorithms to recognize the label of capcha pictures.

### Dataset
#### Original Dataset
Google Recaptcha Image Dataset
https://www.kaggle.com/datasets/mikhailma/test-dataset
Almost 12000 images used in Google Recaptcha V2 collected by category more than 500 of which with manual markup for training object detection model such as YOLO.
One can use those two txt files and Data1.yaml in folder about_datasets to train it with yolo.
#### My own datasets
https://github.com/yiyundotchen/Google-recaptcha-data-for-yolo
##### Dataset1
Using Data1.ipynb in folder about_datasets, I selected useful images from original dataset, and made it suitable for yolo training.
It's smaller than original dataset.
##### Dataset2
As shown in results part, the original data is not good for yolo training. No matter how I tune it, the mAP can't be better than 0.5
So I made labels for bicycle and bus images myself, and made this dataset2 for yolo training.
I used Data2.ipynb in folder about_datasets to adjust this dataset.
One can use Data2.yaml in folder about_datasets to train it with yolo.

### Evaluations
To evaluate object detection models like R-CNN and YOLO, the mean average precision (mAP) is used. The mAP compares the ground-truth bounding box to the detected box and returns a score. The higher the score, the more accurate the model is in its detections.
Yolo training function offers mAP and other evaluations. I put the evaluations of my final result(the best model) in folder best_model.


## Methods
I'm using YOLOv5, which means "You Only Look Once".
Yolo is an algorithm that uses convolutional neural networks for object detection. It improves the detection time given that it predicts objects in real-time and provides accurate results with minimal background errors.
I used yolo both using Jupyter and Google Colab.
When using Jupyter to run yolo, remember to install cuda, so it can be run on GPU instead of CPU, which is much faster.
It's much easier and faster to use Colab. I created more Google accounts so that I can use Colab for free.(Highly recommended!!!!!!!)

## Results
The "img" in forms below means to change the width and height into any number which is divisible by 32.
Those are results for Data1, details are in Data1_using_Jupyter.ipynb and Data1_using_Colab.ipynb in folder trainings and results.
|  img   | batch  |  epochs  |  mAP  |
|  :----:  | :----:  | :----:  | :----:  |
| 120 | 10 | 500 | 0.437 |
| 640 | 2 | 60 | 0.354 |
| 640 | 10 | 60 | 0.36 |
| 640 | 3 | 300 | 0.372 |
| 640 | 10 | 300 | 0.417 |
| 640 | 15 | 300 | 0.444 |
| 480 | 15 | 300 | 0.441 |
| 128 | 15 | 300 | 0.398 |

Those are results for Data2.
|  img   | batch  |  epochs  |  mAP_bicycle  |  mAP_chimney  |  mAP_bus  |  mAP  |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 640 | 2 | 3 | 0.26 | 0.00321 | 0.468 | 0.244 |
| 640 | 16 | 400 | 0.759 | 0.586 | 0.906 | 0.75 |

Those are results for the best model.
![image](best_model/image/results.png)
![image](best_model/image/val_batch1_pred.jpg)

## Prospect
The images used for training are 120*120, which is really unclear. It must have affected the final result.
With those images, my best mAP is 0.763, still lower than 0.8.
I'm planning to preprocess those images to improve the model.


## Reference

https://github.com/ultralytics/yolov5/issues/6916
https://medium.com/analytics-vidhya/understanding-yolo-and-implementing-yolov3-for-object-detection-5f1f748cc63a
https://www.youtube.com/watch?v=GRtgLlwxpc4
https://github.com/ultralytics/yolov5

