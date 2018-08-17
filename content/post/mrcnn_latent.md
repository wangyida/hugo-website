+++
date = 2018-08-15
lastmod = 2018-08-15
draft = false
tags = ["computer vision"]
title = "Modification for MASK-RCNN for Latent Space"
math = true
summary = """
"""

[header]
image = ""
caption = ""

+++

## Examples

``` python
python3 obj_detect_yida.py \
-num_class 90 \
-input_file /media/wangyida/YEEEEDA/videos/1.mp4 \
-output_file /media/wangyida/YEEEEDA/videos/1_result.mp4 \
-model_name faster_rcnn_nas_coco \
-path_ckpt frozen_inference_graph.pb \
-path_label mscoco_label_map.pbtxt
```
<div>
	<table align="center">
		<tr>
		<td>
			<div id="bloc1">
			<video autoplay="autoplay" loop="loop">
				<source src="/img/mrcnn_latent/2.mp4" type="video/mp4" />
			</video>
			</div>
		</td>
		<td>
			<div id="bloc2">
			<video autoplay="autoplay" loop="loop">
				<source src="/img/mrcnn_latent/2_nas.mp4" type="video/mp4" />
			</video>
			</div>
		</td>
		</tr>
	</table>
</div>
		
<div>
	<table align="center">
		<tr>
		<td>
			<div id="bloc1">
			<video autoplay="autoplay" loop="loop">
				<source src="/img/mrcnn_latent/3.mp4" type="video/mp4" />
			</video>
			</div>
		</td>
		<td>
			<div id="bloc2">
			<video autoplay="autoplay" loop="loop">
				<source src="/img/mrcnn_latent/3_nas.mp4" type="video/mp4" />
			</video>
			</div>
		</td>
		</tr>
	</table>
</div>
		
<div>
	<table align="center">
		<tr>
		<td>
			<div id="bloc1">
			<video autoplay="autoplay" loop="loop">
				<source src="/img/mrcnn_latent/4.mp4" type="video/mp4" />
			</video>
			</div>
		</td>
		<td>
			<div id="bloc2">
			<video autoplay="autoplay" loop="loop">
				<source src="/img/mrcnn_latent/4_nas.mp4" type="video/mp4" />
			</video>
			</div>
		</td>
		</tr>
	</table>
</div>
