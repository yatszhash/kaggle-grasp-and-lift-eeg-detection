# my codes for kaggle-grasp-and-lift-eeg-detection

**late challenge** codes for kaggle's grasp and lift eeg detection competition.  

https://www.kaggle.com/c/grasp-and-lift-eeg-detection

## Method

### signal as image with pretrained model

This method is inspired by the paper.

1. Emami, A., Kunii, N., Matsuo, T., Shinozaki, T., Kawai, K., & Takahashi, H. (2019). 
Seizure detection by convolutional neural network-based analysis of scalp electroencephalography plot images. NeuroImage: Clinical, 22, 101684. https://doi.org/10.1016/J.NICL.2019.101684

The method is very simple.  

1. plot all signals for a measurement in the same image (and export into a image file).  
2. fine tune the pre-trained cnn model with ImageNet for the eeg images.  

Please see the paper in detail.  

## result

| local | public LB | private LB|  
|:---|:---|:---|
| 0.94273 |0.49211| 0.55177 |

It seems largely over-fitted to train data.  
That is probably due to data leakage or some mistakes in implementation ...

## TODO 
- [ ] retrain (but currently I don't have enough machine resource)