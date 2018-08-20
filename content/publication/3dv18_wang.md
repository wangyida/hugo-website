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
tags = ["deep-learning", "computer-vision", "generative model", "3D"]
url_code = ""
url_dataset = ""
url_pdf = ""
url_project = "project/deep-learning/"
url_slides = ""
url_video = ""

+++

<video autoplay="autoplay" loop="loop">
	<source src="/img/3dv/3dv_presentation.mp4" type="video/mp4" />
</video>

Our target | Animations
:----:|:----:
<img src="/img/3dv/teaser.png" alt="road condition" width="300" height="300" frameborder="0" style="border:0" > | <img src="/img/3dv/video.gif" alt="road condition" width="300" height="300" frameborder="0" style="border:0" >

## Architecture

![architecture](/img/3dv/architecture.png)
![discriminators](/img/3dv/discriminators.png)

## Qualitative results

![qualitative](/img/3dv/qualitative.png)
