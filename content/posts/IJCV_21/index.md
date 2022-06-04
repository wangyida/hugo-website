---
title: SoftPool++ / An Encoder-Decoder Network for Point Cloud Completion
date: 2021-12-29T10:15:01+02:00
categories: [journal]
tags: [deep learning, computer vision, 3D completion, IJCV]
language: en
cover:
    image: "teasers/thesis_teaser_ijcv22.png"
    alt: 'caption for image'
    caption: "cover image"
slug: youtube
---
> Re-direct to the full [**PAPER**](https://link.springer.com/article/10.1007/s11263-022-01588-7) and [**CODE**]()

# Abstrarct

We propose a novel convolutional operator for the task of point cloud completion. One striking characteristic of our approach is that, conversely to related work it does not require any max-pooling or voxelization operation. Instead, the proposed operator used to learn the point cloud embedding in the encoder extracts permutation-invariant features from the point cloud via a **soft-pooling** of feature activations, which are able to preserve fine-grained geometric details. These features are then passed on to a decoder architecture. Due to the compression in the encoder, a typical limitation of this type of architectures is that they tend to lose parts of the input shape structure. We propose to overcome this limitation by using skip connections specifically devised for point clouds, where links between corresponding layers in the encoder and the decoder are established. As part of these connections, we introduce a transformation matrix that projects the features from the encoder to the decoder and vice-versa. The quantitative and qualitative results on the task of object completion from partial scans on the ShapeNet dataset show that our approach achieves state-of-the-art performance in shape completion both at low and high resolutions.

# Cite

If you find this work useful in your research, please cite:

```bash
@article{wang2022softpool++,
  title={SoftPool++: An Encoder--Decoder Network for Point Cloud Completion},
  author={Wang, Yida and Tan, David Joseph and Navab, Nassir and Tombari, Federico},
  journal={International Journal of Computer Vision},
  pages={1--20},
  year={2022},
  publisher={Springer}
}
```
