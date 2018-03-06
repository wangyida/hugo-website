+++
date = 2018-01-29
lastmod = 2018-03-05
draft = false
tags = ["academic", "python", "annotation"]
title = "基于Mask RCNN的通用物体检测分割平台"
math = true
summary = """
移植mask rcnn进行物体检测、分割、计数。
"""

[header]
image = ""
caption = "Image credit: [**Academic**](https://github.com/gcushen/hugo-academic/)"

+++
# Mask R-CNN 用于对象检测和分割

这是基于Python 3，Keras和TensorFlow上的[Mask R-CNN](https://arxiv.org/abs/1703.06870)的实现。 该模型为图像中的每个对象实例生成边界框和分割掩码。 它基于特征金字塔网络（FPN）和ResNet101主干网。

![实例分割样例](/img/mask_rcnn/street.png)

源代码库包括包括：
* 在FPN和ResNet101上构建的Mask R-CNN的源代码。
* MS COCO的训练代码
* MS COCO预先训练的权重
* Jupyter-notebook可视化脚本
* 用于多GPU并行训练
* 评估MS COCO指标（AP）
* 自定义数据集进行训练的例子


该代码易于扩展。 如果您在研究和工业生产中使用它，能提高工作效率。




## 1. 锚点排序和筛选
可视化第一阶段候选区域网络的每一步，并显示正负锚点以及锚点框架细化。
![](/img/mask_rcnn/detection_anchors.png)

## 2. 边界框细化
这是第二阶段最终检测框（虚线）和应用于它们的细化（实线）的示例。
![](/img/mask_rcnn/detection_refinement.png)

## 3. 掩模生成
生成掩模的实例。 然后将它们缩放并放置在正确位置的图像上。
![](/img/mask_rcnn/detection_masks.png)

## 4. 层激活
通常检查不同层的激活来追踪不合理激活（全零或随机噪声）是有用的。
![](/img/mask_rcnn/detection_activations.png)

## 5. 权重直方图
![](/img/mask_rcnn/detection_histograms.png)

## 6. 将不同的部分组合成最终结果
![](/img/mask_rcnn/detection_final.png)


## 更多实例
![羊](/img/mask_rcnn/sheep.png)
![椰子](/img/mask_rcnn/donuts.png)

# Mask R-CNN 演示实例

快速介绍如何使用预先训练的模型来检测和分割对象。


```python
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

%matplotlib inline 

# 項目的根目錄
ROOT_DIR = os.getcwd()

# 目錄保存日誌和訓練好的模型
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# 訓練的權重文件的本地路徑
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/img/mask_rcnn_coco.h5")
# 如果需要，從远端下載COCO訓練的權重
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# 要運行檢測的圖像目錄
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
```

## 配置

我们将使用在MS-COCO数据集上训练的模型。这个模型的配置位于coco.py```的```CocoConfig``类中。

对于特定的推测任务来说，稍微修改一下配置以适应任务。为此，子类```CocoConfig``应该被你需要改变的属性覆盖。


```python
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
```

    
    Configurations:
    BACKBONE_SHAPES                [[256 256]
     [128 128]
     [ 64  64]
     [ 32  32]
     [ 16  16]]
    BACKBONE_STRIDES               [4, 8, 16, 32, 64]
    BATCH_SIZE                     1
    BBOX_STD_DEV                   [ 0.1  0.1  0.2  0.2]
    DETECTION_MAX_INSTANCES        100
    DETECTION_MIN_CONFIDENCE       0.5
    DETECTION_NMS_THRESHOLD        0.3
    GPU_COUNT                      1
    IMAGES_PER_GPU                 1
    IMAGE_MAX_DIM                  1024
    IMAGE_MIN_DIM                  800
    IMAGE_PADDING                  True
    IMAGE_SHAPE                    [1024 1024    3]
    LEARNING_MOMENTUM              0.9
    LEARNING_RATE                  0.002
    MASK_POOL_SIZE                 14
    MASK_SHAPE                     [28, 28]
    MAX_GT_INSTANCES               100
    MEAN_PIXEL                     [ 123.7  116.8  103.9]
    MINI_MASK_SHAPE                (56, 56)
    NAME                           coco
    NUM_CLASSES                    81
    POOL_SIZE                      7
    POST_NMS_ROIS_INFERENCE        1000
    POST_NMS_ROIS_TRAINING         2000
    ROI_POSITIVE_RATIO             0.33
    RPN_ANCHOR_RATIOS              [0.5, 1, 2]
    RPN_ANCHOR_SCALES              (32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE              2
    RPN_BBOX_STD_DEV               [ 0.1  0.1  0.2  0.2]
    RPN_TRAIN_ANCHORS_PER_IMAGE    256
    STEPS_PER_EPOCH                1000
    TRAIN_ROIS_PER_IMAGE           128
    USE_MINI_MASK                  True
    USE_RPN_ROIS                   True
    VALIDATION_STEPS               50
    WEIGHT_DECAY                   0.0001
    
    


## 创建模型并加载训练后的权重


```python
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
```

## 类名称

该模型对对象进行分类并返回类ID，这是识别每个类的整数值。有些数据集将整数值分配给它们的类，有些则不会。例如，在MS-COCO数据集中，“人”类为1，“泰迪熊”为88.这些ID通常是连续的，但并非总是如此。例如，COCO数据集具有与类别ID 70和72相关联的类别，但不具有类别71。

为了提高一致性，并同时支持来自多个源的数据的训练，我们的Dataset类为每个类分配了自己的顺序整数ID。例如，如果您使用我们的Dataset类加载COCO数据集，那么'person'类将获得class ID = 1（就像COCO），'teddy bear'class是78（与COCO不同） 。在将类ID映射到类名时请记住这一点。

要获得类名的列表，你需要加载数据集，然后使用像这样的```class_names```属性。
```
# Load COCO dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()

# Print class names
print(dataset.class_names)
```

我们不想要求你下载COCO数据集来运行这个演示，所以我们在下面列出了类名的列表。列表中类名称的索引表示其ID（第一类为0，第二类为1，第三类为2，...等）。


```python
# COCO类名
# 列表中类的索引是它的ID。例如，要获得ID的ID
# 泰迪熊课，请使用： class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
```

## 运行对象检测


```python
# 从图像文件夹加载一个随机图像
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# 运行物体检测
results = model.detect([image], verbose=1)

# 可视化结果
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
```

    Processing 1 images
    image                    shape: (476, 640, 3)         min:    0.00000  max:  255.00000
    molded_images            shape: (1, 1024, 1024, 3)    min: -123.70000  max:  120.30000
    image_metas              shape: (1, 89)               min:    0.00000  max: 1024.00000



![png](/img/mask_rcnn/output_9_1.png)


# Mask R-CNN - 训练形状数据集

这里展示了如何在你自己的数据集上训练Mask R-CNN。为了简单起见，我们使用了可以快速训练的形状（正方形，三角形和圆形）的综合数据集。不过，你仍然需要一个GPU，因为网络主干是一个Resnet101，这对于在CPU上训练来说太慢了。在GPU上，您可以在几分钟内开始好的结果，并在不到一个小时的时间内获得好的结果。

形状数据集的代码包含在下面。它可以即时生成图像，因此不需要下载任何数据。它可以生成任何大小的图像，所以我们选择一个小图像大小来加快训练速度。


```python
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

%matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
```

    Using TensorFlow backend.


## 配置


```python
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()
```

    
    Configurations:
    BACKBONE_SHAPES                [[32 32]
     [16 16]
     [ 8  8]
     [ 4  4]
     [ 2  2]]
    BACKBONE_STRIDES               [4, 8, 16, 32, 64]
    BATCH_SIZE                     8
    BBOX_STD_DEV                   [ 0.1  0.1  0.2  0.2]
    DETECTION_MAX_INSTANCES        100
    DETECTION_MIN_CONFIDENCE       0.5
    DETECTION_NMS_THRESHOLD        0.3
    GPU_COUNT                      1
    IMAGES_PER_GPU                 8
    IMAGE_MAX_DIM                  128
    IMAGE_MIN_DIM                  128
    IMAGE_PADDING                  True
    IMAGE_SHAPE                    [128 128   3]
    LEARNING_MOMENTUM              0.9
    LEARNING_RATE                  0.002
    MASK_POOL_SIZE                 14
    MASK_SHAPE                     [28, 28]
    MAX_GT_INSTANCES               100
    MEAN_PIXEL                     [ 123.7  116.8  103.9]
    MINI_MASK_SHAPE                (56, 56)
    NAME                           SHAPES
    NUM_CLASSES                    4
    POOL_SIZE                      7
    POST_NMS_ROIS_INFERENCE        1000
    POST_NMS_ROIS_TRAINING         2000
    ROI_POSITIVE_RATIO             0.33
    RPN_ANCHOR_RATIOS              [0.5, 1, 2]
    RPN_ANCHOR_SCALES              (8, 16, 32, 64, 128)
    RPN_ANCHOR_STRIDE              2
    RPN_BBOX_STD_DEV               [ 0.1  0.1  0.2  0.2]
    RPN_TRAIN_ANCHORS_PER_IMAGE    256
    STEPS_PER_EPOCH                100
    TRAIN_ROIS_PER_IMAGE           32
    USE_MINI_MASK                  True
    USE_RPN_ROIS                   True
    VALIDATION_STEPS               50
    WEIGHT_DECAY                   0.0001
    
    


## 基本绘图参数设置


```python
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
```

## 数据集

创建一个合成数据集

扩展Dataset类并添加一个方法来加载形状数据集`load_shapes()`，并覆盖以下方法：

* load_image()
* load_mask()
* image_reference()


```python
class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes
```


```python
# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()
```


```python
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
```


![png](/img/mask_rcnn/output_19_0.png)



![png](/img/mask_rcnn/output_19_1.png)



![png](/img/mask_rcnn/output_19_2.png)



![png](/img/mask_rcnn/output_19_3.png)


## 创建模型


```python
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
```


```python
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
```

## 训练

分两个阶段训练：
1.只有头部。在这里，我们冻结所有的骨干层，只训练随机初始化层（即我们没有使用MS COCO预先训练的权重）。为了仅训练头层，将`layers ='heads'`传递给`train（）`函数。

2.微调所有图层。对于这个简单的例子，这不是必要的，但我们将其包括在内以显示过程。只需传递`layers =“all`来训练所有图层。


```python
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
```


```python
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")
```


```python
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)
```

## 检测


```python
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
```


```python
# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))
```

    original_image           shape: (128, 128, 3)         min:  108.00000  max:  236.00000
    image_meta               shape: (12,)                 min:    0.00000  max:  128.00000
    gt_bbox                  shape: (2, 5)                min:    2.00000  max:  102.00000
    gt_mask                  shape: (128, 128, 2)         min:    0.00000  max:    1.00000



![png](/img/mask_rcnn/output_29_1.png)



```python
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())
```

    Processing 1 images
    image                    shape: (128, 128, 3)         min:  108.00000  max:  236.00000
    molded_images            shape: (1, 128, 128, 3)      min:  -15.70000  max:  132.10000
    image_metas              shape: (1, 12)               min:    0.00000  max:  128.00000



![png](/img/mask_rcnn/output_30_1.png)


## 验证结果


```python
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
```

    mAP:  0.95


# Mask R-CNN  - 检查训练数据

检查并可视化数据加载和预处理代码。


```python
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

%matplotlib inline 

ROOT_DIR = os.getcwd()
```

    Using TensorFlow backend.


## 配置

运行下面的代码块之一来导入和加载要使用的配置。


```python
# Run one of the code blocks

# Shapes toy dataset
# import shapes
# config = shapes.ShapesConfig()

# MS COCO Dataset
import coco
config = coco.CocoConfig()
COCO_DIR = "path to COCO dataset"  # TODO: enter value here
```

## 数据集


```python
# 加载数据集
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, "train")

# 在使用数据集之前必须调用
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
```

## 显示样本

加载并显示原始图像和掩模图像。


```python
# 加载并显示随机样本
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
```


![png](/img/mask_rcnn/output_40_0.png)



![png](/img/mask_rcnn/output_40_1.png)



![png](/img/mask_rcnn/output_40_2.png)



![png](/img/mask_rcnn/output_40_3.png)


## 边界框

我们不是使用源数据集提供的边界框坐标，而是使用掩码来计算边界框。这使得我们无论源数据集如何都能够一致地处理边界框，并且还可以更轻松地调整大小，旋转或裁剪图像，因为我们只是从更新掩码生成边界框，而不是计算每种图像类型的边界框转换转型。


```python
# 加载并显示图片和掩模
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)

# 计算分割框
bbox = utils.extract_bboxes(mask)

# 显示图片和结果
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)

# 显示图片和独立的物体
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
```

    image_id  74886 http://cocodataset.org/#explore?id=118535
    image                    shape: (375, 500, 3)         min:    0.00000  max:  255.00000
    mask                     shape: (375, 500, 5)         min:    0.00000  max:    1.00000
    class_ids                shape: (5,)                  min:    1.00000  max:   35.00000
    bbox                     shape: (5, 4)                min:    1.00000  max:  329.00000



![png](/img/mask_rcnn/output_42_1.png)


## 调整图像大小

为了支持每批次的多个图像，图像被调整为一个尺寸（1024x1024）。纵横比保存，但。如果图像不是正方形，则会在顶部/底部或右侧/左侧添加零填充。


```python
# Load random image and mask.
image_id = np.random.choice(dataset.image_ids, 1)[0]
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
original_shape = image.shape
# Resize
image, window, scale, padding = utils.resize_image(
    image, 
    min_dim=config.IMAGE_MIN_DIM, 
    max_dim=config.IMAGE_MAX_DIM,
    padding=config.IMAGE_PADDING)
mask = utils.resize_mask(mask, scale, padding)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id: ", image_id, dataset.image_reference(image_id))
print("Original shape: ", original_shape)
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
```

    /usr/local/lib/python3.5/dist-packages/scipy/ndimage/interpolation.py:600: UserWarning: From scipy 0.13.0, the output shape of zoom() is calculated with round() instead of int() - for these inputs the size of the returned array has changed.
      "the returned array has changed.", UserWarning)


    image_id:  6480 http://cocodataset.org/#explore?id=402563
    Original shape:  (476, 640, 3)
    image                    shape: (1024, 1024, 3)       min:    0.00000  max:  255.00000
    mask                     shape: (1024, 1024, 32)      min:    0.00000  max:    1.00000
    class_ids                shape: (32,)                 min:    1.00000  max:   77.00000
    bbox                     shape: (32, 4)               min:    1.00000  max:  991.00000



![png](/img/mask_rcnn/output_44_2.png)


## 迷你掩模图像

使用高分辨率图像进行训练时，实例二进制蒙版可能会变大。例如，如果使用1024x1024映像训练，那么单个实例的掩码需要1MB的内存（Numpy使用字节作为布尔值）。如果一个图像有100个实例，那么只有100MB的掩码。

为了提高训练速度，我们通过以下方式优化口罩：
*我们存储对象边界框内的蒙版像素，而不是完整图像的蒙版。大多数物体与图像大小相比都很小，所以我们通过不在物体周围存储大量零来节省空间。
*我们将面罩调整为较小的尺寸（例如56x56）。对于大于所选大小的对象，我们会失去一点准确性。但是大多数对象注释的开头并不是很精确，所以对于大多数实际用途来说，这种损失是可以忽略的。可以在配置类中设置mini_mask的大小。

为了可视化掩码大小调整的效果，并验证代码的正确性，我们可以看到一些示例。


```python
image_id = np.random.choice(dataset.image_ids, 1)[0]
image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    dataset, config, image_id, use_mini_mask=False)

log("image", image)
log("image_meta", image_meta)
log("class_ids", class_ids)
log("bbox", bbox)
log("mask", mask)

display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
```

    image                    shape: (1024, 1024, 3)       min:    0.00000  max:  255.00000
    image_meta               shape: (89,)                 min:    0.00000  max: 23221.00000
    bbox                     shape: (1, 5)                min:   62.00000  max:  578.00000
    mask                     shape: (1024, 1024, 1)       min:    0.00000  max:    1.00000



![png](/img/mask_rcnn/output_46_1.png)



```python
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
```


![png](/img/mask_rcnn/output_47_0.png)



```python
# Add augmentation and mask resizing.
image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    dataset, config, image_id, augment=True, use_mini_mask=True)
log("mask", mask)
display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
```

    mask                     shape: (56, 56, 1)           min:    0.00000  max:    1.00000



![png](/img/mask_rcnn/output_48_1.png)



```python
mask = utils.expand_mask(bbox, mask, image.shape)
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
```


![png](/img/mask_rcnn/output_49_0.png)


## 锚点

锚的顺序很重要。在训练和预测阶段使用相同的顺序。它必须匹配卷积执行的顺序。

对于一个FPN网络来说，定位锚点的方式必须能够很容易地将锚点与预测锚点分数和移位的卷积层输出相匹配。
*先按金字塔等级排序。第一级的所有锚点，然后是第二级的所有锚点，依此类推。这使得通过级别分离锚更容易。
*在每个级别中，按功能图处理顺序排序锚点。通常，卷积层处理从左上角开始并逐行向右移动的特征图。
*对于每个特征贴图单元格，为不同比例的锚点选择任何排序顺序。这里我们匹配传递给函数的比率顺序。

** 锚点跨度：**
在FPN架构中，前几层的特征映射是高分辨率的。例如，如果输入图像是1024x1024，那么第一层的特征meap是256x256，这会产生大约200K的锚（256 * 256 * 3）。这些锚点是32x32像素，它们相对于图像像素的步幅是4像素，所以有很多重叠。如果我们为特征映射中的每个其他单元生成锚，我们可以显着减少负载。例如，2的步幅会将锚的数量减少4。

在这个实现中，我们使用2的锚定步幅，这与纸张不同。


```python
# Generate Anchors
anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, 
                                          config.RPN_ANCHOR_RATIOS,
                                          config.BACKBONE_SHAPES,
                                          config.BACKBONE_STRIDES, 
                                          config.RPN_ANCHOR_STRIDE)

# Print summary of anchors
num_levels = len(config.BACKBONE_SHAPES)
anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
print("Count: ", anchors.shape[0])
print("Scales: ", config.RPN_ANCHOR_SCALES)
print("ratios: ", config.RPN_ANCHOR_RATIOS)
print("Anchors per Cell: ", anchors_per_cell)
print("Levels: ", num_levels)
anchors_per_level = []
for l in range(num_levels):
    num_cells = config.BACKBONE_SHAPES[l][0] * config.BACKBONE_SHAPES[l][1]
    anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
    print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))
```

Visualize anchors of one cell at the center of the feature map of a specific level.


```python
## Visualize anchors of one cell at the center of the feature map of a specific level

# Load and draw random image
image_id = np.random.choice(dataset.image_ids, 1)[0]
image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)
levels = len(config.BACKBONE_SHAPES)

for level in range(levels):
    colors = visualize.random_colors(levels)
    # Compute the index of the anchors at the center of the image
    level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
    level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
    print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0], 
                                                                config.BACKBONE_SHAPES[level]))
    center_cell = config.BACKBONE_SHAPES[level] // 2
    center_cell_index = (center_cell[0] * config.BACKBONE_SHAPES[level][1] + center_cell[1])
    level_center = center_cell_index * anchors_per_cell 
    center_anchor = anchors_per_cell * (
        (center_cell[0] * config.BACKBONE_SHAPES[level][1] / config.RPN_ANCHOR_STRIDE**2) \
        + center_cell[1] / config.RPN_ANCHOR_STRIDE)
    level_center = int(center_anchor)

    # Draw anchors. Brightness show the order in the array, dark to bright.
    for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
        y1, x1, y2, x2 = rect
        p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
                              edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
        ax.add_patch(p)

```

    /usr/local/lib/python3.5/dist-packages/scipy/ndimage/interpolation.py:600: UserWarning: From scipy 0.13.0, the output shape of zoom() is calculated with round() instead of int() - for these inputs the size of the returned array has changed.
      "the returned array has changed.", UserWarning)


    Level 0. Anchors:  49152  Feature map Shape: [256 256]
    Level 1. Anchors:  12288  Feature map Shape: [128 128]
    Level 2. Anchors:   3072  Feature map Shape: [64 64]
    Level 3. Anchors:    768  Feature map Shape: [32 32]
    Level 4. Anchors:    192  Feature map Shape: [16 16]



![png](/img/mask_rcnn/output_53_2.png)


## 数据生成器



```python
# Create data generator
random_rois = 2000
g = modellib.data_generator(
    dataset, config, shuffle=True, random_rois=random_rois, 
    batch_size=4,
    detection_targets=True)
```


```python
# Uncomment to run the generator through a lot of images
# to catch rare errors
# for i in range(1000):
#     print(i)
#     _, _ = next(g)
```


```python
# Get Next Image
if random_rois:
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
    [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
    
    log("rois", rois)
    log("mrcnn_class_ids", mrcnn_class_ids)
    log("mrcnn_bbox", mrcnn_bbox)
    log("mrcnn_mask", mrcnn_mask)
else:
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)
    
log("gt_class_ids", gt_class_ids)
log("gt_boxes", gt_boxes)
log("gt_masks", gt_masks)
log("rpn_match", rpn_match, )
log("rpn_bbox", rpn_bbox)
image_id = image_meta[0][0]
print("image_id: ", image_id, dataset.image_reference(image_id))

# Remove the last dim in mrcnn_class_ids. It's only added
# to satisfy Keras restriction on target shape.
mrcnn_class_ids = mrcnn_class_ids[:,:,0]
```


```python
b = 0

# Restore original image (reverse normalization)
sample_image = modellib.unmold_image(normalized_images[b], config)

# Compute anchor shifts.
indices = np.where(rpn_match[b] == 1)[0]
refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
log("anchors", anchors)
log("refined_anchors", refined_anchors)

# Get list of positive anchors
positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
print("Positive anchors: {}".format(len(positive_anchor_ids)))
negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
print("Negative anchors: {}".format(len(negative_anchor_ids)))
neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

# ROI breakdown by class
for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
    if n:
        print("{:23}: {}".format(c[:20], n))

# Show positive anchors
visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids], 
                     refined_boxes=refined_anchors)
```

    anchors                  shape: (65472, 4)            min: -362.03867  max: 1258.03867
    refined_anchors          shape: (4, 4)                min:  112.99997  max:  912.00000
    Positive anchors: 4
    Negative anchors: 252
    Neutral anchors: 65216
    BG                     : 90
    chair                  : 6
    bed                    : 30
    remote                 : 2



![png](/img/mask_rcnn/output_58_1.png)



```python
# Show negative anchors
visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])
```


![png](/img/mask_rcnn/output_59_0.png)



```python
# Show neutral anchors. They don't contribute to training.
visualize.draw_boxes(sample_image, boxes=anchors[np.random.choice(neutral_anchor_ids, 100)])
```


![png](/img/mask_rcnn/output_60_0.png)


## ROIs


```python
if random_rois:
    # Class aware bboxes
    bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]

    # Refined ROIs
    refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:,:4] * config.BBOX_STD_DEV)

    # Class aware masks
    mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]

    visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)
    
    # Any repeated ROIs?
    rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
    _, idx = np.unique(rows, return_index=True)
    print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))
```

    Positive ROIs:  38
    Negative ROIs:  90
    Positive Ratio: 0.30
    Unique ROIs: 128 out of 128



![png](/img/mask_rcnn/output_62_1.png)



```python
if random_rois:
    # Dispalay ROIs and corresponding masks and bounding boxes
    ids = random.sample(range(rois.shape[1]), 8)

    images = []
    titles = []
    for i in ids:
        image = visualize.draw_box(sample_image.copy(), rois[b,i,:4].astype(np.int32), [255, 0, 0])
        image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
        images.append(image)
        titles.append("ROI {}".format(i))
        images.append(mask_specific[i] * 255)
        titles.append(dataset.class_names[mrcnn_class_ids[b,i]][:20])

    display_images(images, titles, cols=4, cmap="Blues", interpolation="none")
```


![png](/img/mask_rcnn/output_63_0.png)



```python
# Check ratio of positive ROIs in a set of images.
if random_rois:
    limit = 10
    temp_g = modellib.data_generator(
        dataset, config, shuffle=True, random_rois=10000, 
        batch_size=1, detection_targets=True)
    total = 0
    for i in range(limit):
        _, [ids, _, _] = next(temp_g)
        positive_rois = np.sum(ids[0] > 0)
        total += positive_rois
        print("{:5} {:5.2f}".format(positive_rois, positive_rois/ids.shape[1]))
    print("Average percent: {:.2f}".format(total/(limit*ids.shape[1])))
```
