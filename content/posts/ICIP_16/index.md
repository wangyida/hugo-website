---
title: Self-restraint Object Recognition by Model Based CNN Learning
date: 2016-04-01T10:15:01+02:00
categories: [conference]
tags: [deep learning, computer vision, object detection, ICIP]
language: en
slug: icip16
---

# Abstrarct

CNN has shown excellent performance on object recognition based on huge amount of real images. For training with synthetic data rendered from 3D models alone to reduce the workload of collecting real images, we propose a concatenated self-restraint learning structure lead by a triplet and softmax jointed loss function for object recognition. Locally connected auto encoder trained from rendered images with and without background used for object reconstruction against environment variables produces an additional channel automatically concatenated to RGB channels as input of classification network. This structure makes it possible training a softmax classifier directly from CNN based on synthetic data with our rendering strategy. Our structure halves the gap between training based on real photos and 3D model in both PASCAL and ImageNet database compared to GoogleNet.

# Cite

If you find this work useful in your research, please cite:

```bash
@inproceedings{wang2016self,
  title={Self-restraint object recognition by model based CNN learning},
  author={Wang, Yida and Deng, Weihong},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  pages={654--658},
  year={2016},
  organization={IEEE}
}
```
