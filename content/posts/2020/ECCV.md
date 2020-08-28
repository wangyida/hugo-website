---
title: SoftPoolNet - Shape Descriptor for Point Cloud Completion and Classification
date: 2020-08-25T10:15:01+02:00
categories: [publication]
tags: [machine learning, demo, computer vision]
language: en
slug: youtube
---

{{< youtube zw4NlyxWlBg >}}

If you find this work useful in yourr research, please cite:

```bash
@article{DBLP:journals/corr/abs-2008-07358,
  author    = {Yida Wang and
               David Joseph Tan and
               Nassir Navab and
               Federico Tombari},
  title     = {SoftPoolNet: Shape Descriptor for Point Cloud Completion and Classification},
  journal   = {CoRR},
  volume    = {abs/2008.07358},
  year      = {2020}
}
```

# Abstrarct

Point clouds are often the default choice for many applications as they exhibit more flexibility and efficiency than volumetric data. Nevertheless, their unorganized nature -- points are stored in an unordered way -- makes them less suited to be processed by deep learning pipelines. In this paper, we propose a method for 3D object completion and classification based on point clouds. We introduce a new way of organizing the extracted features based on their activations, which we name soft pooling. For the decoder stage, we propose regional convolutions, a novel operator aimed at maximizing the global activation entropy. Furthermore, inspired by the local refining procedure in Point Completion Network (PCN), we also propose a patch-deforming operation to simulate deconvolutional operations for point clouds. This paper proves that our regional activation can be incorporated in many point cloud architectures like AtlasNet and PCN, leading to better performance for geometric completion. We evaluate our approach on different 3D tasks such as object completion and classification, achieving state-of-the-art accuracy.
