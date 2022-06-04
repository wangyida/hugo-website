---
title: Learning Local Displacements for Point Cloud Completion
date: 2022-02-19T10:15:01+02:00
categories: [conference]
tags: [deep learning, computer vision, 3D completion, semantic completion, CVPR, point cloud]
language: en
cover:
    image: "teasers/thesis_teaser_cvpr22.png"
    alt: 'caption for image'
    caption: "Semantic scene completion results on the CompleteScanNet dataset"
---

> Re-direct to the full [**PAPER**](https://arxiv.org/pdf/2203.16600v1.pdf) and [**CODE**](https://github.com/wangyida/disp3d) 

{{< youtube -rSLpHYO78M >}}

# Abstrarct
| Completing a car |  |
| :-: | :-- |
![teaser](images/CVPR_teaser.png#center) | From the input partial scan to our object completion, we visualize the amount of detail in our reconstruction.

We propose a novel approach aimed at object and semantic scene completion from a partial scan represented as a 3D point cloud.
Our architecture relies on three novel layers that are used successively within an encoder-decoder structure and specifically developed for the task at hand.
The first one carries out feature extraction by matching the point features to a set of pre-trained local descriptors.
Then, to avoid losing individual descriptors as part of standard operations such as max-pooling, we propose an alternative neighbor-pooling operation that relies on adopting the feature vectors with the highest activations. Finally, up-sampling in the decoder modifies our feature extraction in order to increase the output dimension.
While this model is already able to achieve competitive results with the state of the art, we further propose a way to increase the versatility of our approach to process point clouds. To this aim, we introduce a second model that assembles our layers within a transformer architecture.
We evaluate both architectures on object and indoor scene completion tasks, achieving state-of-the-art performance.

# Methodology
## Local displacement operator
| The operation |  |
| :-: | :-- |
![operator](images/CVPR_graph_conv.png#center) | (a) *k*-nearest neighbor in reference to an anchor **f**; (b) displacement vectors around the anchor **f** + δ<sub>i</sub> and the corresponding weight σ<sub>i</sub>; and, (c) closest features for all i.

## Architectures
| The *direct* architectrue | The *transformer* architecture |
| :-: | :-: |
![direct](images/CVPR_direct_architecture.png#center) | ![transformer](images/CVPR_transformer_architecture.png#center)

## Qualitatives
### Object completion
![objects](images/CVPR_shapenet.png#center)

### Semantic scene completion
![objects](images/CVPR_scannet.png#center)

# Cite

If you find this work useful in your research, please cite:

```bash
@inproceedings{wang2022displacement,
  title={Learning Local Displacements for Point Cloud Completion},
  author={Wang, Yida and Tan, David Joseph and Navab, Nassir and Tombari, Federico},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
