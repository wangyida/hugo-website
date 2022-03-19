---
title: SoftPoolNet - Shape Descriptor for Point Cloud Completion and Classification
date: 2020-08-25T10:15:01+02:00
categories: [conference]
tags: [deep learning, computer vision, 3D completion, ECCV]
language: en
cover:
    image: "teasers/thesis_teaser_eccv20.png"
    alt: 'caption for image'
    caption: "cover image"
slug: youtube
---
| [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480069.pdf) | [code](https://github.com/wangyida/softpool) |

{{< youtube zw4NlyxWlBg >}}

# Abstrarct

Point clouds are often the default choice for many applications as they exhibit more flexibility and efficiency than volumetric data. Nevertheless, their unorganized nature -- points are stored in an unordered way -- makes them less suited to be processed by deep learning pipelines. In this paper, we propose a method for 3D object completion and classification based on point clouds. We introduce a new way of organizing the extracted features based on their activations, which we name soft pooling. For the decoder stage, we propose regional convolutions, a novel operator aimed at maximizing the global activation entropy. Furthermore, inspired by the local refining procedure in Point Completion Network (PCN), we also propose a patch-deforming operation to simulate deconvolutional operations for point clouds. This paper proves that our regional activation can be incorporated in many point cloud architectures like AtlasNet and PCN, leading to better performance for geometric completion. We evaluate our approach on different 3D tasks such as object completion and classification, achieving state-of-the-art accuracy.

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
