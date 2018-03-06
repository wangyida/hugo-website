+++
date = 2017-11-01
lastmod = 2017-11-01
draft = false
tags = ["shell"]
title = "Shell Tips"
math = true
summary = """
Shell tips.
"""

[header]
image = ""
caption = "Image credit: [**Academic**](https://github.com/gcushen/hugo-academic/)"

+++

## File Transmitting 

```bash
scp -P 10639 /Users/yidawang/Documents/gitfarm/cluster-vae/list_attr_celeba.csv user@557803.iask.in:/home/user/Desktop/
```

```bash
rsync -avz --progress -e 'ssh -p 10639' /Users/yidawang/Downloads/img_align_celeba.zip user@557803.iask.in:/home/user/Desktop/
```
