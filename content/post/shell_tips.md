+++
date = 2017-11-01
lastmod = 2017-11-01
draft = false
tags = ["shell"]
title = "Shell Tips"
math = true
summary = """
Shell tips for image processing.
"""

[header]
image = ""
caption = "Image credit: [**Academic**](https://github.com/gcushen/hugo-academic/)"

+++

## Image processing

### Trim images in batches

#### Images

Suppose that there are images in `tmp` folder, you can remove all margins without any effective pixel by executing:

```bash
find tmp -name 'scatter*.png' -print0 | xargs -0 -I {} convert {} -trim {}
```

Option `print0` for `find` is mean to print the full file name on the standard output, 
followed by a null character (instead of the newline character that `-print` uses).  
This allows file names that contain newlines or other types of white space to be correctly interpreted by programs that process the find output. 
This option corresponds to the `-0` option of `xargs`.

If you want to turn white background into a transparency one in form of `png` images, `-fuzz 2% -transparent white` should be added.
Suppose that all images you want to trim and remove white backgrounds, the command should be something like:

```bash
find . -name '*.png' -print0 | xargs -0 -I {} convert {} -trim -fuzz 2% -transparent white {}
```
#### PDF files

```bash
for FILE in ./*.pdf; do pdfcrop --margins '5 5 5 5' "${FILE}"; done
```

### Append images

Vertically:
```bash
convert *.png -append -fuzz 2% -transparent white diff_ours.png
```

Horizontally:
```bash
convert *.png +append -fuzz 2% -transparent white diff_ours.png
```

Those command will keep all original images' own shapes and barely concatenate them in RGB space

### Concatenate images to a video

Firstly, we should rename files in the form `foo1, foo2, ..., foo1300, ..., fooN` to form `foo00001, foo00002, ..., foo01300, ..., fooN` by:
```bash
for (( i=0; i<2000; i++ )); do printf -v n "%04d" $i; mv "$i.png" "$n.png"; done;
```

Then use ffmpeg for video producing:
```bash
ffmpeg -framerate 50 -pattern_type glob -i "*.png" -s:v 640x480 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p hand_pose.mp4
```

Sometimes it's necessary to embed small scale videos on websitem, so GIF files should be generated from original videos in format of `.mp4`, `.mov` or something else.
Firstly let's observe the effect of the transparency mechanism introduced in 2013 in the GIF encoder:
```bash
ffmpeg -v warning -ss 00:00:00.000 -t 00:00:38.000 -i hand_pose.mp4 -vf scale=300:-1 -gifflags -transdiff -y bbb-notrans.gif
```
```bash
ffmpeg -v warning -ss 00:00:00.000 -t 00:00:38.000 -i hand_pose.mp4 -vf scale=300:-1 -gifflags +transdiff -y bbb-trans.gif
```

No trans | With trans
:-------------------------:|:-------------------------:
![](/img/bbb-notrans.gif) | ![](/img/bbb-trans.gif)

GIF is limited to a palette of 256 colors. And by default, FFmpeg just uses a generic palette that tries to cover the whole color space in order to support the largest variety of content. [Here](http://blog.pkh.me/p/21-high-quality-gif-with-ffmpeg.html) is a good post for generating high quality `.gif` files with compact structure.

In order to circumvent this problem, dithering is used. In the Big Buck Bunny GIF above, the ordered bayer dithering is applied. It is easily recognizable by its 8x8 crosshatch pattern. While it's not the prettiest, is has many benefits such as being predictible, fast, and actually preventing banding effects and similar visual glitches.
```bash
ffmpeg -v warning -ss 00:00:00.000 -t 00:00:38.000 -i hand_pose.mp4 -vf scale=300:-1:sws_dither=ed -y bbb-error-diffusal.gif
```

The second pass is then handled by the paletteuse filter, which as you guessed from the name will use that palette to generate the final color quantized stream. Its task is to find the most appropriate color in the generated palette to represent the input color. This is also where you will decide which dithering method to use.
```bash
ffmpeg -v warning -ss 00:00:00.000 -t 00:00:38.000 -i hand_pose.mp4 -vf scale=300:-1:lanczos -y bbb-lanczos.gif
```

Error diffusal | Lanczos 
:-------------------------:|:-------------------------:
![](/img/bbb-error-diffusal.gif) | ![](/img/bbb-lanczos.gif)

The file sizes are:

* -rw-r--r--@ 1 yidawang  staff    24M Dec 14 17:06 bbb-error-diffusal.gif
* -rw-r--r--@ 1 yidawang  staff   5.8M Dec 14 17:09 bbb-lanczos.gif
* -rw-r--r--@ 1 yidawang  staff    21M Dec 14 15:33 bbb-notrans.gif
* -rw-r--r--@ 1 yidawang  staff   5.8M Dec 14 15:34 bbb-trans.gif
* -rw-r--r--@ 1 yidawang  staff   3.4M Dec 14 17:26 bbb.mp4

## File Transmitting 

```bash
scp -P 10639 /Users/yidawang/Documents/gitfarm/cluster-vae/list_attr_celeba.csv user@557803.iask.in:/home/user/Desktop/
```

```bash
rsync -avz --progress -e 'ssh -p 10639' /Users/yidawang/Downloads/img_align_celeba.zip user@557803.iask.in:/home/user/Desktop/
```
