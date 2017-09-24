+++
abstract = "Given large amount of real photos for training, Convolutional neural network shows excellent performance on object recognition tasks. However, the process of collecting data is so tedious and the background are also limited which makes it hard to establish a perfect database. In this paper, our generative model trained with synthetic images rendered from 3D models reduces the workload of data collection and limitation of conditions. Our structure is composed of two sub-networks: semantic foreground object reconstruction network based on Bayesian inference and classification network based on multi-triplet cost function for avoiding over-fitting problem on monotone surface and fully utilizing pose information by establishing sphere-like distribution of descriptors in each category which is helpful for recognition on regular photos according to poses, lighting condition, background and category information of rendered images. Firstly, our conjugate structure called generative model with metric learning utilizing additional foreground object channels generated from Bayesian rendering as the joint of two sub-networks. Multi-triplet cost function based on poses for object recognition are used for metric learning which makes it possible training a category classifier purely based on synthetic data. Secondly, we design a coordinate training strategy with the help of adaptive noises acting as corruption on input images to help both sub-networks benefit from each other and avoid inharmonious parameter tuning due to different convergence speed of two sub-networks. Our structure achieves the state of the art accuracy of over 50 percent on ShapeNet database with data migration obstacle from synthetic images to real photos. This pipeline makes it applicable to do recognition on real images only based on 3D models."
abstract_short = "Our structure is composed of two sub-networks: semantic foreground object reconstruction network based on Bayesian inference and classification network based on multi-triplet cost function for avoiding over-fitting problem on monotone surface and fully utilizing pose information by establishing sphere-like distribution of descriptors in each category which is helpful for recognition on regular photos according to poses, lighting condition, background and category information of rendered images. It achieves the state of the art accuracy of over 50 percent on ShapeNet database with data migration obstacle from synthetic images to real photos. This pipeline makes it applicable to do recognition on real images only based on 3D models."
authors = ["Yida Wang, Weihong Deng"]
date = "2017-05-01"
image_preview = "pipeline_tip.svg"
math = true
publication_types = ["3"]
publication = "In *Image Processing (TIP) peer review*, IEEE."
publication_short = "TIP peer review"
selected = true
title = "Generative Model with Coordinate Metric Learning for Object Recognition Based on 3D Models"
url_code = "https://github.com/wangyida/gm-cml"
url_dataset = "https://shapenet.cs.stanford.edu/"
url_pdf = "https://arxiv.org/pdf/1705.08590.pdf"
url_project = "project/deep-learning/"
url_slides = ""
url_video = ""

+++

![pipeline_tip](pipeline_tip.svg)
![imganalysis](imganalysis.svg)
![triplet_samples](triplet_samples.svg)
More detail can easily be written here using *Markdown* and $\rm \LaTeX$ math code.
