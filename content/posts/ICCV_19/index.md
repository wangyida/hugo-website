---
title: ForkNet, Multi-Branch Volumetric Semantic Completion From a Single Depth Image
date: 2019-11-01T10:15:01+02:00
categories: [conference]
tags: [deep learning, computer vision, semantic completion, adversarial training, ICCV, volumetric data]
language: en
cover:
    image: "teasers/thesis_teaser_iccv19.png"
    alt: 'caption for image'
    caption: "semantic scene completion"
slug: iccv19
---
> Re-direct to the full [**PAPER**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_ForkNet_Multi-Branch_Volumetric_Semantic_Completion_From_a_Single_Depth_Image_ICCV_2019_paper.pdf) and [**CODE**](https://github.com/wangyida/forknet)

{{< youtube 1WZ16bGff1o >}}

# Abstrarct

| Scene completion | Object completion |
| :-: | :-: |
![direct](images/ForkNet_nyu.gif#center) | ![transformer](images/ForkNet_shapenet.gif#center)

We propose a novel model for 3D semantic completion from a single depth image, based on a single encoder and three separate generators used to reconstruct different geometric and semantic representations of the original and completed scene, all sharing the same latent space. To transfer information between the geometric and semantic branches of the network, we introduce paths between them concatenating features at corresponding network layers.  Motivated by the limited amount of training samples from real scenes, an interesting attribute of our architecture is the capacity to supplement the existing dataset by generating a new training dataset with high quality, realistic scenes that even includes occlusion and real noise. We build the new dataset by sampling the features directly from latent space which generates a pair of partial volumetric surface and completed volumetric semantic surface. Moreover, we utilize multiple discriminators to increase the accuracy and realism of the reconstructions. We demonstrate the benefits of our approach on standard benchmarks for the two most common completion tasks: semantic 3D scene completion and 3D object completion.

# Methodology
## Architecture
![direct](images/architecture.png#center)
The proposed volumetric network architecture for semantic completion relies on a shared latent space encoded from SDF volume *x* reconstructed from the input depth image. The two decoding paths are trained to generate, respectively, incomplete surface geometry (*x*ˆ), completed geometric volume (*g*) and completed semantic volumes (*s*).

![direct](images/micro_architecture.png#center)
(a-b) Downsam- pling and (c) upsampling convolutional layers in our architecture. Note that the two parameters (*s, d*) in all the functions are the stride and dilation while the kernel size is set to 3.

## Learning
![direct](images/path.png#center)
Graphical models of the 4 data flows (and the associated loss terms) used during training and derived from our architecture figure.

# Solving practical problems
## Solving lack of paired training data
| Synthetic paired data | |
| :-: | :-: |
![direct](images/learning_dataset.png#center) | The synthetically generated SDF volume and the corresponding completed semantic scene.

## Correcting wrong annotations
| Annotation correction | |
| :-: | :-: |
![direct](images/incorrect_annotation.png#center) | ForkNet corrects the wrong ground truth from SUNCG dataset on the TVs.

# Qualitatives
## Semantic scene completion
![direct](images/qualitative.png#center)

## Object completion
![direct](images/qualitative_obj.png#center)

# Cite

If you find this work useful in your research, please cite:

```bash
@inproceedings{wang2019forknet,
  title={ForkNet: Multi-branch Volumetric Semantic Completion from a Single Depth Image},
  author={Wang, Yida and Tan, David Joseph and Navab, Nassir and Tombari, Federico},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8608--8617},
  year={2019}
}
```
