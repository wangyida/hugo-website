+++
abstract = "One of the bottlenecks in acquiring a perfect database for deep learning is the tedious process of collecting and labeling data. In this paper, we propose a generative model trained with synthetic images rendered from 3D models which can reduce the burden on collecting real training data and make the background conditions more realistic. Our architecture is composed of two sub-networks: a semantic foreground object reconstruction network based on Bayesian inference, and a classification network based on multi-triplet cost training for avoiding over-fitting on the monotone synthetic object surface and utilizing accurate information of synthetic images like object poses and lighting conditions which are helpful for recognizing regular photos. Firstly, our generative model with metric learning utilizes additional foreground object channels generated from semantic foreground object reconstruction sub-network for recognizing the original input images. Multi-triplet cost function based on poses is used for metric learning which makes it possible to train an effective categorical classifier purely based on synthetic data. Secondly, we design a coordinate training strategy with the help of adaptive noise applied on the inputs of both of the concatenated sub-networks to make them benefit from each other and avoid inharmonious parameter tuning due to different convergence speed of two sub-networks. Our architecture achieves the state of the art accuracy of 50.5% on the ShapeNet database with data migration obstacle from synthetic images to real images. This pipeline makes it applicable to do recognition on real images only based on 3D models. Our codes are available at Github"
authors = ["Yida Wang, Weihong Deng"]
date = "2018-06-01"
image_preview = "pipeline_tip.svg"
math = true
publication_types = ["2"]
publication = "In *IEEE Transactions on Image Processing*, IEEE"
publication_short = "TIP"
selected = true
title = "Generative Model with Coordinate Metric Learning for Object Recognition Based on 3D Models"
tags = ["deep-learning", "variational-inference", "computer-vision"]
url_code = "https://github.com/wangyida/gm-cml"
url_dataset = "https://shapenet.cs.stanford.edu/"
url_pdf = "https://arxiv.org/abs/1705.08590"
url_project = "project/deep-learning/"
url_slides = ""
url_video = ""

+++
input | target | manifold 
:----:|:----:|:----: 
![test_xs](/img/test_xs.png) | ![test_ts](/img/test_ts.png) | ![manifold_latest](/img/manifold_latest.png) 

Method pipeline.
![pipeline_tip](/img/pipeline_tip.png)

Image analysis for average and standard deviation for data released together with our paper.
![imganalysis](/img/imganalysis.svg)

Examples for triplet set used in our metric learning.
![triplet_samples](/img/triplet_samples.svg)

4 nearest neighboour retrieval results based on simple models.
![triplet_demo](/img/publication/tip/tip_triplet.gif)

## Examples for real world tasks

### Depth prediction
Example for depth prediction based on RGB images.
![show](/img/show.gif)

### Scene understanding 
Examples for videos collected by [Bleenco](https://bleenco.com/).
The first columns are reference videos and the second columns are videos shot by another camera which is set to be the prediction results based on the reference videos.
The third columns are the prediction results where the scenes are predicted staticlly based on the scene shot by the second camera and people are predicted dynamically based on the understanding for input videos.

The predicted people are mostly blurred because the architecture is robust to changes of clothes.

**TRAINING**
Scene 1 in Bleenco

![](/img/bleenco_scene_1_train.gif)

Scene 2 in Bleenco

![](/img/bleenco_scene_2_train.gif)

Scene 3 in Bleenco

![](/img/bleenco_scene_3_train.gif)

Scene 4 in Bleenco

![](/img/bleenco_scene_4_train.gif)

**VALIDATING**
Scene 1 in Bleenco

![](/img/bleenco_scene_1_valid.gif)

Scene 2 in Bleenco

![](/img/bleenco_scene_2_valid.gif)

Scene 3 in Bleenco

![](/img/bleenco_scene_3_valid.gif)

Scene 4 in Bleenco

![](/img/bleenco_scene_4_valid.gif)

