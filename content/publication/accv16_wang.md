+++
abstract = "Effective utilization on texture-less 3D models for deep learning is significant to recognition on real photos. We eliminate the reliance on massive real training data by modifying convolutional neural network in 3 aspects: synthetic data rendering for training data generation in large quantities, multi-triplet cost function modification for multi-task learning and compact micro architecture design for producing tiny parametric model while overcoming over-fit problem in texture-less models. Network is initiated with multi-triplet cost function establishing sphere-like distribution of descriptors in each category which is helpful for recognition on regular photos according to pose, lighting condition, background and category information of rendered images. Fine-tuning with additional data further meets the aim of classification on special real photos based on initial model. We propose a 6.2 MB compact parametric model called ZigzagNet based on SqueezeNet to improve the performance for recognition by applying moving normalization inside micro architecture and adding channel wise convolutional bypass through macro architecture. Moving batch normalization is used to get a good performance on both convergence speed and recognition accuracy. Accuracy of our compact parametric model in experiment on ImageNet and PASCAL samples provided by PASCAL3D+ based on simple Nearest Neighbor classifier is close to the result of 240 MB AlexNet trained with real images. Model trained on texture-less models which consumes less time for rendering and collecting outperforms the result of training with more textured models from ShapeNet."
abstract_short = "We eliminate the reliance on massive real training data by modifying convolutional neural network in 3 aspects: synthetic data rendering for training data generation in large quantities, multi-triplet cost function modification for multi-task learning and compact micro architecture design for producing tiny parametric model while overcoming over-fit problem in texture-less models."
authors = ["Yida Wang, Can Cui, Xiuzhuang Zhou, Weihong Deng"]
date = "2017-03-12"
image_preview = "paperArch-eps-converted-to.svg"
math = true
publication_types = ["6"]
publication = "In *Asian Conference on Computer Vision (ACCV 2016)*, Springer."
publication_short = "In *ACCV*"
selected = true 
title = "ZigzagNet: Efficient Deep Learning for Real Object Recognition Based on 3D Models"
url_code = ""
url_dataset = ""
url_pdf = "https://www.researchgate.net/publication/314635459_ZigzagNet_Efficient_Deep_Learning_for_Real_Object_Recognition_Based_on_3D_Models"
url_project = "project/deep-learning/"
url_slides = ""
url_video = ""

+++

![paperArch-eps-converted-to](paperArch-eps-converted-to.svg)
![micro_ours-eps-converted-to](micro_ours-eps-converted-to.svg)
More detail can easily be written here using *Markdown* and $\rm \LaTeX$ math code.
