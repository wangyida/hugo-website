---
title: ForkNet - Multi-Branch Volumetric Semantic Completion From a Single Depth Image
date: 2019-11-01T10:15:01+02:00
categories: [conference]
tags: [deep learning, computer vision, 3D completion, adversarial training, ICCV]
language: en
cover:
    image: "teasers/thesis_teaser_iccv19.png"
    alt: 'caption for image'
    caption: "cover image"
slug: youtube
---
| [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_ForkNet_Multi-Branch_Volumetric_Semantic_Completion_From_a_Single_Depth_Image_ICCV_2019_paper.pdf) | [code](https://github.com/wangyida/forknet) |

{{< youtube 1WZ16bGff1o >}}

# Abstrarct

We propose a novel model for 3D semantic completion from a single depth image, based on a single encoder and three separate generators used to reconstruct different geometric and semantic representations of the original and completed scene, all sharing the same latent space. To transfer information between the geometric and semantic branches of the network, we introduce paths between them concatenating features at corresponding network layers.  Motivated by the limited amount of training samples from real scenes, an interesting attribute of our architecture is the capacity to supplement the existing dataset by generating a new training dataset with high quality, realistic scenes that even includes occlusion and real noise. We build the new dataset by sampling the features directly from latent space which generates a pair of partial volumetric surface and completed volumetric semantic surface. Moreover, we utilize multiple discriminators to increase the accuracy and realism of the reconstructions. We demonstrate the benefits of our approach on standard benchmarks for the two most common completion tasks: semantic 3D scene completion and 3D object completion.

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
