---
title: Generative Model with Coordinate Metric Learning for Object Recognition Based on 3D Models
date: 2017-11-15T10:15:01+02:00
categories: [journal]
tags: [deep learning, computer vision, variational inference, TIP]
language: en
slug: tip17
---

# Abstrarct

Collecting data for deep learning is so tedious which makes it hard to establish a perfect database. In this paper, we propose a generative model trained with synthetic images rendered from 3D models which can reduce the burden on collecting real training data and make the background conditions more sundry. Our architecture is composed of two subnetworks: semantic foreground object reconstruction network based on Bayesian inference and classification network based on multi-triplet cost training for avoiding over-fitting on monotone synthetic object surface and utilizing accurate informations of synthetic images like object poses and lightning conditions which are helpful for recognizing regular photos. Firstly, our generative model with metric learning utilizes additional foreground object channels generated from semantic foreground object reconstruction sub-network for recognizing the original input images.  Multi-triplet cost function based on poses is used for metric learning which makes it possible training an effective categorical classifier purely based on synthetic data. Secondly, we design a coordinate training strategy with the help of adaptive noises applied on inputs of both of the concatenated sub-networks to make them benefit from each other and avoid inharmonious parameter tuning due to different convergence speed of two subnetworks. Our architecture achieves the state of the art accuracy of 50.5% on ShapeNet database with data migration obstacle from synthetic images to real photos. This pipeline makes it applicable to do recognition on real images only based on 3D models.

# Cite

If you find this work useful in your research, please cite:

```bash
@article{wang2018generative,
  title={Generative Model With Coordinate Metric Learning for Object Recognition Based on 3D Models},
  author={Wang, Yida and Deng, Weihong},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={12},
  pages={5813--5826},
  year={2018},
  publisher={IEEE}
}
```

