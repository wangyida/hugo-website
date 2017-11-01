+++
date = 2017-11-01
lastmod = 2017-11-01
draft = false
tags = ["shell"]
title = "Shell Tips"
math = true
summary = """
"""

[header]
image = ""
caption = "Image credit: [**Academic**](https://github.com/gcushen/hugo-academic/)"

+++

## Trim images in batches

Suppose that there are images in `tmp` folder, you can remove all margins without any effective pixel by executing:

```bash
find tmp -name 'scatter*.png' -print0 | xargs -0 -I {} convert {} -trim {}
```

option `print0` for `find` is mean to print the full file name on the standard output, 
followed by a null character (instead of the newline character that `-print` uses).  
This allows file names that contain newlines or other types of white space to be correctly interpreted by programs that process the find output. 
This option corresponds to the `-0` option of xargs.
