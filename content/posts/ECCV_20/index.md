---
title: SoftPoolNet, Shape Descriptor for Point Cloud Completion and Classification
date: 2020-08-25T10:15:01+02:00
categories: [conference]
tags: [deep learning, computer vision, 3D completion, ECCV, point cloud]
language: en
cover:
    image: "teasers/thesis_teaser_eccv20.png"
    alt: 'caption for image'
    caption: "completed bi-wings plane"
---
> Re-direct to the full [**PAPER**](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480069.pdf) and [**CODE**](https://github.com/wangyida/softpool) 

{{< youtube zw4NlyxWlBg >}}

# Abstrarct

Point clouds are often the default choice for many applications as they exhibit more flexibility and efficiency than volumetric data. Nevertheless, their unorganized nature -- points are stored in an unordered way -- makes them less suited to be processed by deep learning pipelines. In this paper, we propose a method for 3D object completion and classification based on point clouds. We introduce a new way of organizing the extracted features based on their activations, which we name soft pooling. For the decoder stage, we propose regional convolutions, a novel operator aimed at maximizing the global activation entropy. Furthermore, inspired by the local refining procedure in Point Completion Network (PCN), we also propose a patch-deforming operation to simulate deconvolutional operations for point clouds. This paper proves that our regional activation can be incorporated in many point cloud architectures like AtlasNet and PCN, leading to better performance for geometric completion. We evaluate our approach on different 3D tasks such as object completion and classification, achieving state-of-the-art accuracy.

# Methodology
## SoftPool
![softpool](images/softpool.png#center) 
Toy examples of (a) sorting the the *k*-th element of the vectors in the feature matrix **F** to build **F**′k and consequently **F**′ and (b) concatenation of the first *N* rows of **F**′k to construct the 2D matrix **F**∗ which corresponds to the regions with high activations.

## SoftPoolNet *vs* PointNet
soft-pool *vs* max-pool
![pointnet_compare](images/pointnet.png#center) 
Comparison between (a) our soft-pool operation and (b) max-pooling from PointNet, where the feature from PointNet is only a subset of our feature. 

![pointnet_sample_compare](images/pointnet_compare.png#center) 
Comparison of our method and PointNet where PointNet reconstructs the more typical four-leg table instead of six in (c). 

## Regional subsets
| Colored points from each orders|  |
| :-: | :-- |
![subsets](images/softpool_subsets.png#center) | Results when choosing the first subsets of *N*r with the following ranges [1 : 2], [1 : 4], [1 : 8], [1 : 16] and [1 : 32] when the architecture is trained with *N*r= 32.

# Cite

If you find this work useful in your research, please cite:

```bash
@inproceedings{wang2020softpoolnet,
  title={Softpoolnet: Shape descriptor for point cloud completion and classification},
  author={Wang, Yida and Tan, David Joseph and Navab, Nassir and Tombari, Federico},
  booktitle={European Conference on Computer Vision},
  pages={70--85},
  year={2020},
  organization={Springer}
}
```
