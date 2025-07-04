# YOLACT Real-time Instance Segmentation
## Introduction
YOLACT, which stands for "You Only Look At Coefficients," represents a seminal advancement in the field of computer vision by establishing a new paradigm for real-time instance segmentation that fundamentally re-architects the traditional detection-then-segmentation pipeline. Unlike its two-stage predecessors like Mask R-CNN, YOLACT introduces a one-stage, fully-convolutional approach that operates in parallel: it simultaneously generates a set of instance-agnostic "prototype masks" covering the entire image space while a separate prediction head computes class confidences, bounding boxes, and a unique vector of "mask coefficients" for each detected object. The final, high-quality instance mask is then dynamically assembled in a single, computationally trivial step by linearly combining these prototype masks with the predicted coefficients. This parallel design not only preserves spatial coherence across the image but also eliminates the significant computational bottleneck of feature localization and mask generation, enabling YOLACT to achieve unprecedented real-time performance (over 30 FPS) on a single GPU and thereby setting a new benchmark for solving complex scene understanding tasks under strict latency constraints.

## A. Dataset and Pre-processsing
### Prepare the COCO 2017 TFRecord Dataset
[2017 Train images](http://images.cocodataset.org/zips/train2017.zip)  / [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) / [2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) <br/>

Extract the ```/train2017```, ```/val2017```, and ```/annotations/instances_train2017.json```, ```/annotations/instances_val2017.json ```from annotation to ```./data``` folder of the repo, and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir './data/train2017' 
                                      -val_image_dir './data/val2017' 
                                      -train_annotations_file './data/instances_train2017.json' 
                                      -val_annotations_file './data/instances_val2017.json' 
                                      -output_dir './data/coco'
```

Extract the ```/benchmark/dataset/img ``` folder from benchmark.tgz, and ```pascal_sbd_train.json```, ```pascal_sbd_valid.json``` from annotation to ```./data``` folder of the repo. Divinding images into 2 folders (```/pascal_train``` for training, ```/pascal_val``` for validation images.) and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir './data/pascal_train' 
                                      -val_image_dir './data/pascal_val' 
                                      -train_annotations_file './data/pascal_sbd_train.json' 
                                      -val_annotations_file './data/pascal_sbd_valid.json' 
                                      -output_dir './data/pascal'
```

### Prepare Custom Dataset
Create a folder of training images, a folder of validation images, and a COCO-style annotation like above for your dataset in ```./data``` folder of the repo, and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir 'path to your training images' 
                                      -val_image_dir   'path to your validaiton images'  
                                      -train_annotations_file 'path to your training annotations' 
                                      -val_annotations_file 'path to your validation annotations' 
                                      -output_dir './data/name of the dataset'
```
## Training
### 1. Configuration for COCO, Pascal SBD
The configuration for experiment can be adjust in ```config.py```. The default hyperparameters from original paper are already written as example for you to know how to customize it. You can adjust following parameters:

#### Parameters for Parser
| Parameters | Description |
| --- | --- |
| `NUM_MAX_PAD` | The maximum padding length for batching samples. |
| `THRESHOLD_POS` | The positive threshold iou for anchor mathcing. |
| `THRESHOLD_NEG` | The negative threshold iou for anchor mathcing. |

#### Parameters for Model
| Parameters | Description |
| --- | --- |
| `BACKBONE` | The name of backbone model defined in `backbones_objects` .|
| `IMG_SIZE` | The input size of images.|
| `PROTO_OUTPUT_SIZE` | Output size of protonet.|
| `FPN_CHANNELS` | The Number of convolution channels used in FPN.|
| `NUM_MASK`| The number of predicted masks for linear combination.|

#### Parameters for Loss
| Parameters for Loss | Description |
| --- | --- |
| `LOSS_WEIGHT_CLS` | The loss weight for classification. |
| `LOSS_WEIGHT_BOX` | The loss weight for bounding box. |
| `LOSS_WEIGHT_MASK` | The loss weight for mask prediction. |
| `LOSS_WEIGHT_SEG` | The loss weight for segamentation. |
| `NEG_POS_RATIO` | The neg/pos ratio for OHEM in classification. |

#### Parameters for Detection
| Parameters | Description |
| --- | --- |
| `CONF_THRESHOLD` | The threshold for filtering possible detection by confidence score. |
| `TOP_K` | The maximum number of input possible detection for FastNMS. |
| `NMS_THRESHOLD` | The threshold for FastNMS. |
| `MAX_NUM_DETECTION` | The maximum number of detection.|


### Configuration for Custom Dataset (to be updated)
```bash




```
### Check the Dataset Sample 
```bash




```

### Training Script
-> Training for COCO:
```bash
python train.py -name 'coco'
                -tfrecord_dir './data'
                -weights './weights' 
                -batch_size '8'
                -momentum '0.9'
                -weight_decay '5 * 1e-4'
                -print_interval '10'
                -save_interval '5000'
```
-> Training for Pascal SBD:
```bash
python train.py -name 'pascal'
                -tfrecord_dir './data'
                -weights './weights' 
                -batch_size '8'
                -momentum '0.9'
                -weight_decay '5 * 1e-4'
                -print_interval '10'
                -save_interval '5000'
```
-> Training for custom dataset:
```bash
python train.py -name 'name of your dataset'
                -tfrecord_dir './data'
                -weights 'path to store weights' 
                -batch_size 'batch_size'
                -momentum 'momentum for SGD'
                -weight_decay 'weight_decay rate for SGD'
                -print_interval 'interval for printing training result'
                -save_interval 'interval for evaluation'
```
## Inference (to be updated)
There are serval evaluation scenario.
```bash




```
### Test Detection
```bash




```
### Evaluation
```bash




```
### Images
```bash




```
### Videos 
```bash




```


## Reference
* https://github.com/dbolya/yolact
* https://github.com/leohsuofnthu/Tensorflow-YOLACT/blob/master/data/create_coco_tfrecord.py
* https://github.com/tensorflow/models/blob/master/official/vision/detection/dataloader/retinanet_parser.py
* https://github.com/balancap/SSD-Tensorflow
