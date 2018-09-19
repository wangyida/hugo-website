+++
abstract = "We propose a method to reconstruct, complete and semantically label a 3D scene from a single input depth image. We improve the accuracy of the regressed semantic 3D maps by a novel architecture based on adversarial learning. In particular, we suggest using multiple adversarial loss terms that not only enforce realistic outputs with respect to the ground truth, but also an effective embedding of the internal features. This is done by correlating the latent features of the encoder working on partial 2.5D data with the latent features extracted from a variational 3D auto-encoder trained to reconstruct the complete semantic scene. In addition, differently from other approaches that operate entirely through 3D convolutions, at test time we retain the original 2.5D structure of the input during downsampling to improve the effectiveness of the internal representation of our model. We test our approach on the main benchmark datasets for semantic scene completion to qualitatively and quantitatively assess the effectiveness of our proposal." 
abstract_short = "We propose a method to reconstruct, complete and semantically label a 3D scene from a single input depth image."
authors = ["Yida Wang, David Tan, Nassir Navab and Federico Tombari"]
date = "2018-07-21"
image_preview = "objectrec.svg"
math = true
publication_types = ["1"]
publication = "In *International Conference on 3D Vision*, IEEE."
publication_short = "3DV"
selected = true
title = "Adversarial Semantic Scene Completion from a Single Depth Image"
tags = ["deep learning", "computer vision", "generative model", "3D"]
url_code = "https://github.com/wangyida/gan-depth-semantic3d"
url_dataset = ""
url_pdf = ""
url_project = "project/deep-learning/"
url_slides = ""
url_video = "https://youtu.be/udvBhkupwXE"

+++

# <video autoplay="autoplay" loop="loop">
<video id="video" controls preload="metadata">
	<source src="/img/3dv/presentation.mp4" type="video/mp4" />
	<track label="English" kind="subtitles" srclang="en" src="/img/3dv/presentation.vtt">
</video>

## Overview
![](/img/3dv/overview.png)
We introduce a direct reconstruction method to reconstruct from a 2.5D depth image to a 3D voxel data with both shape completion and semantic segmentation that relies on a deep architecture based on 3D VAE with an adversarial training to improve the performance of this task.

## Architecture
![](/img/3dv/architecture.png)
We utilize the latent representation of 3D auto-encoder to help train a latent representation from a depth image. The 3D auto-encoder is removed after the parametric model is trained. This pipeline is optimized for the encoders for the depth image and the 3D volumetric data and the shared generator is also optimised during
training.

## Discriminators
![](/img/3dv/discriminators.png)
To make the latent representation and the reconstructed 3D scene similar to each others, we apply two discriminators for both targets. In this manner, the latent representation of the depth produces the expected target more precisely compared to the latent representation of the ground truth volumetric data.

## Our data format
![](/img/3dv/data_format.png)

## Qualitative results
![](/img/3dv/qualitative_results.png)
