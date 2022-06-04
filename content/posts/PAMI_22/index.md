---
title: Self-supervised Latent Space Optimization with Nebula Variational Coding
date: 2022-03-01T10:15:01+02:00
categories: [journal]
tags: [deep learning, computer vision, 3D completion, PAMI, pose estimation, planar reconstruction, semantic segmentation, point cloud, machine translation, semantic completion, variational inference, clustering]
language: en
cover:
    image: "teasers/thesis_teaser_pami22.png"
    alt: 'caption for image'
    caption: "cover image"
---
> Re-direct to the full [**PAPER**](https://ieeexplore.ieee.org/document/9740011/) and [**CODE**](https://github.com/wangyida/nebula-anchors)

# Abstrarct

Deep learning approaches process data in a layer-by-layer way with intermediate (or latent) features. We aim at designing a general solution to optimize the latent manifolds to improve the performance on classification, segmentation, completion and/or reconstruction through probabilistic models. This paper proposes a variational inference model which leads to a clustered embedding. We introduce additional variables in the latent space, called *nebula anchors*, that guide the latent variables to form clusters during training. To prevent the anchors from clustering among themselves, we employ the variational constraint that enforces the latent features within an anchor to form a Gaussian distribution, resulting in a generative model we refer as Nebula Variational Coding (NVC). Since each latent feature can be labeled with the closest anchor, we also propose to apply metric learning in a self-supervised way to make the separation between clusters more explicit. As a consequence, the latent variables of our variational coder form clusters which adapt to the generated semantic of the training data, *e.g*. the categorical labels of each sample. We demonstrate experimentally that it can be used within different architectures designed to solve different problems including text sequence, images, 3D point clouds and volumetric data, validating the advantage of our proposed method. 

## Latent distributions
![clusters](images/clusters.png#center)

## Latent covariance
| Covariance matrix of each latent features |  |
| :-: | :-- |
![covariance](images/PAMI_covariance.png#center) | Comparison of covariance matrices computed from the latent features that relies on variational constraint..

# Cite

If you find this work useful in your research, please cite:

```bash
@article{wang2022self,
  title={Self-supervised Latent Space Optimization with Nebula Variational Coding},
  author={Wang, Yida and Tan, David Joseph and Navab, Nassir and Tombari, Federico},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
