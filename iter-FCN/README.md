# Iterative fully convolutional neural networks for automatic vertebra segmentation
Automatic vertebra segmentation from medical images such as CT or MRI is a challenging yet essential task for clinical diagnosis and surgical planning due to the complex anatomy and close proximity of spinal structures. Traditional fully convolutional networks (FCNs) often struggle with accurately segmenting individual vertebrae, especially in 3D volumes where anatomical variability and inter-class similarity are high. To address these challenges, we propose an Iterative Fully Convolutional Neural Network (IterativeFCN) that performs vertebra segmentation in a patch-wise, instance-aware manner. By incorporating both the current image patch and a memory map of previously segmented structures, the model iteratively focuses on one vertebra at a time, enabling improved localization and separation of adjacent vertebrae without the need for explicit instance labels. This approach not only enhances segmentation accuracy but also introduces a scalable solution for robust, automated vertebral analysis in volumetric medical imaging.


## Model
This model contains a similiar shape like [3D U-Net] but with constant channel in every layer and a extend branch for classification propose. There are 2 inputs for this model, inclusing image patch and correspond instanace memory patches. Instance Memory is used to remind the model to segment the first 'unsegmented vertebrae' so as to make sure the vertebrae are segmented one by one.


## Dataset and Pre-processsing

### 1. Dataset
I choose one of the dataset used in the paper, The spine segmentation challenge in CSI2014. The dataset can be obtain in the Dataset 2 posted on [SpineWeb](http://spineweb.digitalimaginggroup.ca/spineweb/index.php?n=Main.Datasets#Dataset_2.3A_Spine_and_Vertebrae_Segmentation)

### 2. Data preprocessing
The preprocessing steps of each CT images and corresponded masks(both train and test set) includes:
* **Resample the images and masks to isotropic (1mm * 1mm * 1mm)**
* **Calculate the weight penalty coefficient for each images via distance transform.**
* **Crop the images and masks to remove the vertebrae that not have labels in masks.**
* **Prepare the training patches, including "image patches", "instance memory patches", "mask patches" and "weight patches".**

### 3. Illustration of training patches.
## Training Detail
I apply the same setting as suggested in papers:
* **Batch-size = 1 due to GPU memory limitation.**
* **Adam with learning rate = 1e-3**
* **Apply data augmentation via elastic deformation, gaussain blur, gaussian noise, random crop along z-axis**
* **Produce empty mask training example every 5th iteratons.**

I trained this model on Google Colab, which has similiar CUDA Memory(12GB) with NVIDIA TITANX. Here is trained only with around 25000 iterations. The initial learning rate at 1e-3 from 1 to 10000 iterations, 1e-4 for 10001 to 20000 and 1e-5 for the rest of iterations, which is different from paper that using 1e-3 for whole training.

## Segmentation Result
The following are some segmentation result from both train and test data.

### (1)Visual Result

### (2)Averge Dice Coefficient 
| Result        | Paper         |
| ------------- | ------------- |
| 0.918         | 0.958         |

P.S. None of refine technique for preprocessing and postprocessing are used in this repo.

## Usage
### Setup the Environment
The requirment.txt are provided in the repo
```bash
pip install -r requirements.txt
```

### Preprocessing the CSI dataset
```bash
python -m data.preprocessing --dataset 'the root path of CSI dataset'
```

### Start Training
```bash
python train.py --dataset 'the directory of preprocessed CSI dataset'
```

### Instance Segmentation 
```bash
python instance_segmentation.py --test_dir 'the directory of test images' --weights 'pretrained weights'
```

### Evaluation the Dice Coefficient with labels
```bash
python eval.py --label_dir 'directory of test labels' --pred_dir 'the directory of prediction segmetnation'
```


## Reference
* https://www.youtube.com/watch?v=0we-WooGqxw
* https://gist.github.com/erniejunior/601cdf56d2b424757de5
* https://github.com/SimpleITK/SimpleITK/issues/561
