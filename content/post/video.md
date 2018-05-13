+++
date = 2017-11-01
lastmod = 2017-11-01
draft = false
tags = ["shell", "video"]
title = "Video Processing"
math = true
summary = """
Shell tips for video processing.
"""

[header]
image = ""
caption = "Image credit: [**Academic**](https://github.com/gcushen/hugo-academic/)"

+++

## Concatenate images to a video

Firstly, we should rename files in the form `foo1, foo2, ..., foo1300, ..., fooN` to form `foo00001, foo00002, ..., foo01300, ..., fooN` by:
```bash
for (( i=0; i<2000; i++ )); do printf -v n "%04d" $i; mv "$i.png" "$n.png"; done;
```

Then use ffmpeg for video producing:
```bash
ffmpeg -framerate 50 -pattern_type glob -i "*.png" -s:v 640x480 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p hand_pose.mp4
```
## Video compressing

Given a video in format like `.mp4` and `.mov`, you can compress it using:
```sh
ffmpeg -i panda.mp4 -vcodec h264 -acodec aac panda_small.mp4
```
Here the optional codecs (decoders and encoders): `-vcodec` and `-acodec`.

## From video to GIF

Sometimes it's necessary to embed small scale videos on websitem, so GIF files should be generated from original videos in format of `.mp4`, `.mov` or something else.
Firstly let's observe the effect of the transparency mechanism introduced in 2013 in the GIF encoder:
```bash
ffmpeg -v warning -ss 00:00:00.000 -t 00:00:38.000 -i hand_pose.mp4 -vf scale=300:-1 -gifflags -transdiff -y bbb-notrans.gif
```
```bash
ffmpeg -v warning -ss 00:00:00.000 -t 00:00:38.000 -i hand_pose.mp4 -vf scale=300:-1 -gifflags +transdiff -y bbb-trans.gif
```

GIF is limited to a palette of 256 colors. And by default, FFmpeg just uses a generic palette that tries to cover the whole color space in order to support the largest variety of content. [Here](http://blog.pkh.me/p/21-high-quality-gif-with-ffmpeg.html) is a good post for generating high quality `.gif` files with compact structure.

In order to circumvent this problem, dithering is used. In the Big Buck Bunny GIF above, the ordered bayer dithering is applied. It is easily recognizable by its 8x8 crosshatch pattern. While it's not the prettiest, is has many benefits such as being predictible, fast, and actually preventing banding effects and similar visual glitches.
```bash
ffmpeg -v warning -ss 00:00:00.000 -t 00:00:38.000 -i hand_pose.mp4 -vf scale=300:-1:sws_dither=ed -y bbb-error-diffusal.gif
```

The second pass is then handled by the paletteuse filter, which as you guessed from the name will use that palette to generate the final color quantized stream. Its task is to find the most appropriate color in the generated palette to represent the input color. This is also where you will decide which dithering method to use.
```bash
ffmpeg -v warning -ss 00:00:00.000 -t 00:00:38.000 -i hand_pose.mp4 -vf scale=300:-1:lanczos -y bbb-lanczos.gif
```

No trans | With trans | Error diffusal | Lanczos 
:---:|:---:|:---:|:---:
![](/img/bbb-notrans.gif) | ![](/img/bbb-trans.gif) | ![](/img/bbb-error-diffusal.gif) | ![](/img/bbb-lanczos.gif)

## Images(PNG, BMP, JPEG...) to GIF

Here is a way presenting several related series of frames spatially reasonable in form of GIF. Video is not necessary as a bridge between collection of images and GIF where `convert` is used as basic image processing tool.

```bash
# Define a main function
main()
{
	folder=$1 
	for f in ${folder}input_*.png 
	do 
		convert $f -size 10x xc:none ${f/input/target} -size 10x xc:none ${f/input/recon} \
			+append ${f/input/merged} 
	done 
	convert -delay 15 -loop 0 ${folder}merged_*.png \
		${folder}bleenco_scene.gif
}
```

example on bleenco dataset
```bash
# Scanning for each folders and make the animation
main result_bleenco_1/generated/
main result_bleenco_2/generated/
main result_bleenco_3/generated/
main result_bleenco_4/generated/
```

Scene 1 in Bleenco

![](/img/bleenco_scene_1_train.gif)

Scene 2 in Bleenco

![](/img/bleenco_scene_2_train.gif)

Scene 3 in Bleenco

![](/img/bleenco_scene_3_train.gif)

Scene 4 in Bleenco

![](/img/bleenco_scene_4_train.gif)

