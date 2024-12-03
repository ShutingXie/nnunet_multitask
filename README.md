# Multi-Task nnUNet: Joint Segmentation and Classification for Medical Imaging
This repository contains the official implementation of "Multi-Task nnUNet: Extending nnUNet for Joint Segmentation and Classification Tasks". This work builds upon the well-established nnUNet framework, introducing modifications to enable multi-task learning.


## Overview

Our method integrates segmentation and classification tasks into a single architecture, effectively improving efficiency and accuracy in medical image analysis workflows. Below is a summary of the key features:

Segmentation: Accurate delineation of anatomical structures.
Classification: Predicting pathological subtypes alongside segmentation.
Multi-task Learning: Joint optimization of segmentation and classification for better generalization.

If you use this code or find it helpful, please also cite nnUNet:
```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```


## Environments and Requirements

This implementation is designed to run on **Google Colab Pro**, which provides a pre-configured environment with the following specifications:

- **OS**: Linux (Ubuntu 18.04 or higher)
- **CPU**: 8 cores (Google Colab Pro)
- **RAM**: 15GB (Google Colab Pro)
- **GPU**: NVIDIA Tesla K80/T4/P100 (varies by Colab session type), I used T4
- **CUDA**: Version pre-installed in Colab (verified automatically), I used 12.1
- **Python**: Version 3.10

To set up the environment in Google Colab:
```bash
git clone https://github.com/your-repo/multitask-nnunet.git
cd multitask-nnunet
pip install -e .
```

To set environment variables in Google Colab:
```bash
import os
os.environ["nnUNet_raw"] = "/content/drive/MyDrive/<your_project_name>/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/content/drive/MyDrive/<your_proect_name>/UNet_preprocessed"
os.environ["nnUNet_results"] = "/content/drive/MyDrive/<your_project_name>/nnUNet_rawnnUNet_results"
```

You can always execute `echo ${nnUNet_raw}` etc to print the environment variables. This will return an empty string if they were not set.


## Dataset

This project uses the nnUNet framework, which has specific requirements for dataset structure. For detailed instructions, please refer to the [nnUNet Dataset Format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).


## Preprocessing

The preprocessing of the dataset is handled by nnUNet's automated pipeline, which extracts a dataset fingerprint and prepares preprocessed data for different U-Net configurations. The preprocessing steps include cropping, resampling, and intensity normalization based on the dataset's specific properties.
Running the data preprocessing code:
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlannerResEncM --verify_dataset_integrity
```
- -d 001: Dataset ID. Replace 001 with your dataset's unique ID.
- -pl nnUNetPlannerResEncM: Selects a customized planner, please refer to [Residual Encoder Presets in nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)
- --verify_dataset_integrity: Checks for dataset consistency.

For more options, run:
```bash
nnUNetv2_plan_and_preprocess -h
```


## Training

1. To train the model(s) in the paper, run this command:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerMultitask -p nnUNetResEncUNetMPlans
```


## Inference
TBA


## Results
Our method achieves the following performance:

| Model name             |  DICE  |
| ---------------------- | :----: |
| nnUNetTrainerMultitask | 62.69% |





   
