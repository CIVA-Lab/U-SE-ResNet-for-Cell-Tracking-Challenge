# U-SE-ResNet
The U-SE-ResNet is the network used for Cell Tracking Challenge 2021. Our cell segmentation pipeline consists of three main modules: image preprocessing, cell segmentation using a custom deep convolutional U-Net, named U-SE-ResNet similar to our previous Motion U-Net having a deeper backbone with squeeze-and-excitation, and post-processing. 

<br/>

![](/figures/main-pipeline.png)

<br/>

## Pre-processing
We apply standard normalization with 1% intensity clipping after we convert the 16-bit input to 32-bit float. This would remove outlier pixels and improve the contrast of images. This normalization step is the same for all 2D and 3D datasets. As a data augmentation strategy for training our networks (explained in SEGMENTATION), we have applied a random 512x512 crop (translation), random rotation, random flipping, and random gaussian noise. 

<br/>

![](/figures/pre-processing.png)

<br/>

## Segmentation
We use a custom deep convolutional U-Net, named U-SE-ResNet similar to our previous Motion U-Net to generate segmentation cell masks and cell markers, where cell markers are an erosion of the cells. The erosion size is different for each dataset that is denoted by E-p. 

<br/>

![](/figures/mask-marker.png)

<br/>

U-SE-ResNet is an encoder-decoder type architecture with a ResNet-50 backbone used for feature extraction in the encoder module. Each of the residual blocks in the encoder are equipped with a Squeeze and Excitation blocks. The overall network is similar to the U-Net architecture with skip connections after each block of ResNet-50. 

<br/>

![](/figures/U-SE-ResNet-Arch.png)

<br/>

Since we want to optimize the overlap score, a binary cross entropy loss was used to train the network. We give a weight to each pixel in our final loss map before we aggregate; this weight is higher around edges of a cell and lower away from the cells. This way, we guide our network to learn better cell outlines. 

<br/>

![](/figures/wieghted-loss.png)

<br/>

Our U-SE-ResNet was trained for 120 epochs with mini-batch size of 4. Adam optimizer was used during training with an initial learning rate of 1e-4 that was reduced by a factor of 2 after every 30 epochs. For all of the datasets we evaluated, we used only the provided silver/gold truth masks to train the network, with a split of 90% for training and 10% for validation.

<br/>

## Post-processing
The described network produces two output maps. The first output map is a binary cells mask, the second is a binary mask for cell markers. We use connected components algorithm on the markers to generate the different cell IDs. Then, we use the distance transform of the binary cell mask, and we use that along with the labeled marker map to generate a final labeled cell mask. In order to remove small spurious detections mathematical morphology operations were used. Spurious detections were eliminated by removing connected components in the cell masks of size smaller than T-a.

```Link of this method in the Cell Tracking Challenge:``` http://celltrackingchallenge.net/participants/MU-Ra-US/ 

<br/>

# How to train and test U-SE-ResNet

```SW``` folder contains all scripts used to train and test models. More details on how to train and test the code is mentioned inside ```SW``` folder.

Train and test data should be placed inside ```Data``` folder. More details about data hierarchy is mentioned inside ```Data``` folder.

<br/>

# Project Collaborators and Contact

**Author:** Gani Rahmon, Imad Eddine Toubal and Kannappan Palaniappan

Copyright &copy; 2021-2022. Gani Rahmon, Imad Eddine Toubal and Dr. K. Palaniappan and Curators of the University of Missouri, a public corporation. All Rights Reserved.

**Created by:** Ph.D. students: Gani Rahmon and Imad Eddine Toubal
Department of Electrical Engineering and Computer Science,  
University of Missouri-Columbia  

For more information, contact:

* **Gani Rahmon**  
226 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
gani.rahmon@mail.missouri.edu  

* **Imad Eddine Toubal**  
226 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211   
itoubal@mail.missouri.edu

* **Dr. K. Palaniappan**  
205 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
palaniappank@missouri.edu
