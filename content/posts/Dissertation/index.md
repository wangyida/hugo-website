---
title: Learning-based Single View Completion
date: 2023-05-22T10:15:01+02:00
categories: [dissertation]
tags: [deep learning, computer vision, 3D completion, single view, dissertation]
language: en
cover:
    image: "teasers/teaser_dissertation.png"
    alt: 'caption for image'
    caption: "An exemplar semantic completion from a 3D partial scan in (a) presented in point cloud which is back- projected from a single depth image with camera intrinsic, and the expected semantically completed 3D scene in (b)."
---

> Re-direct to my [**dissertation site**](https://mediatum.ub.tum.de/?id=1653291) and [**manuscript**](https://mediatum.ub.tum.de/doc/1653291/1653291.pdf)


# Abstrarct
Fed with a partial scan, single-view 3D completion models are expected to generate the completed geometry as an inference result. Applications such as AR/VR and autonomous driving could benefit from targets being presented in their complete structures, while occluded geometries are missing due to limited camera view. Learning-based 3D completion approach can reveal the occluded regions conditioned on a single view sample with the help of large-scale training data.

It is straightforward to adapt existing image-space methods to design encoder-decoder architectures constructed with convolutions in volumetric space to process local and global information. Nonetheless, the slow inference speed limits the practicality of the volumetric completion. The computation and memory costs increase exponentially regarding the resolution because of the convolutions. Apart from the methodology designs, the lack of a large amount of real 3D supervision, because of the difficulty in annotating, also makes it hard to optimize models in volumetric space for real scenarios.

Due to the memory cost, the volumetric completion is struggling to reconstruct finer details practically. Such difficulty motivated us further to investigate the works on a sparse data format, the point cloud, for 3D completion. Since the point cloud does not present information in empty spaces, such a sparse 3D representation can be with flexible local resolution depending on specific local geometric complexity. However, a straightforward adaptation with a similar encoder-decoder architecture as the volumetric methods does not work because of the unorganized structure of point cloud. This situation implies that the well-known operators employed on images or volumes cannot be applied to point cloud.

Therefore, in this dissertation, we propose solutions to address all the aforementioned problems in both volumetric data and point cloud for the task of 3D completion from a single view. Notably, while other point cloud methods are limited to reconstructing a synthetic single object, our proposed approach alleviated this limitation by reconstructing real scenes. Inspired by the improvements from all the completion methods, we also demonstrate their generalizability by expanding their applications to other tasks, e.g., hand pose estimation, point cloud segmentation, and natural language processing.



# Cite

If you find this work useful in your research, please cite:

```bash
@phdthesis{dissertation,
  author = {Wang, Yida},
  title = {Learning-based Single View 3D Completion},
  year = {2023},
  school = {Technische Universität München},
  pages = {213},
  language = {en},
  abstract = {Understanding 3D from 2D is crucial for augmented reality and virtual reality applications such as 3D virtual mapping and autonomous driving. In this dissertation, both dense and sparse 3D data formats are investigated for the topic of 3D semantic completion. We highlight that in this dissertation, our proposed inference models can do semantic completion for both volumetric data and point cloud data.},
  keywords = {3D completion, semantic completion, single view},
  note = {},
}
```
