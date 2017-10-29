+++
date = 2017-10-29
lastmod = 2017-10-29
draft = false
tags = ["academic", "python", "annotation"]
title = "labelme: Image Annotation Tool with Python"
math = true
summary = """
"""

[header]
image = ""
caption = "Image credit: [**Academic**](https://github.com/gcushen/hugo-academic/)"

+++
<img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/FCN1.png" align="right" />

Labelme is a graphical image annotation tool inspired by <http://labelme.csail.mit.edu>.  
It is written in Python and uses Qt for its graphical interface.


Requirements
------------

- Ubuntu / macOS / Windows
- Python2 / Python3
- [PyQt4 / PyQt5](http://www.riverbankcomputing.co.uk/software/pyqt/intro)


Installation
------------

There are options:

- Platform agonistic installation: Anaconda, Docker
- Platform specific installation: Ubuntu, macOS

**Anaconda**

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```bash
conda create --name=labelme python=2.7
source activate labelme
conda install pyqt
pip install labelme
```

**Docker**

You need install [docker](https://www.docker.com), then run below:

```bash
wget https://raw.githubusercontent.com/wkentaro/labelme/master/scripts/labelme_on_docker
chmod u+x labelme_on_docker

# Maybe you need http://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/ on macOS
./labelme_on_docker static/apc2016_obj3.jpg -O static/apc2016_obj3.json
```

**Ubuntu**

```bash
sudo apt-get install python-qt4 pyqt4-dev-tools
sudo pip install labelme
```

**macOS**

*Install Homebrew*
	
Homebrew is a package manager for OS X. A package is a collection of code files that work together. Installing them usually means running a script (a bit of code) that puts certain files in the various directories. A lot of the packages you will want are going to have dependencies. That means they require you to have other packages already installed on your computer. Homebrew will find and install dependencies for you AND it will keep them organized in one location AND it can tell you when updates are available for them.  On top of all of that it gives super helpful instructions when everything doesn't go smoothly. You can read more about it at Homebrew's website. For now, install Homebrew using the following line of code:

```bash	
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

```
	
So what's going on here? The last bit is obviously a URL. If you were to open this URL in your browser, you would just see code. This is a Ruby script that tells your computer what to do to install Homebrew. The `curl` part is a command line tool that transfers files using URLs. The `-fsSL` part is a combination of four option flags for `curl` that specify how to handle the file at the url. If you want to know more about what these flags do, type `man curl` at your command prompt. (You can use `man` in front of most commands to open up a manual page for that command.) We also need to actually execute this Ruby script, so we used the command `ruby` at the beginning. The `-e` is an option flag for ruby that executes a string as one line of code, in this case, the `"$(curl … /go)"` part. You may need to follow a few more instructions to finish the install, but Homebrew will help you do so.

*Install Python*
	
Python comes with OS X, so you can probably don't need to do this step. You can check this by typing `python --version` into Terminal. If you get an error message, you need to install Python. If Terminal prints something like `Python 2.7.3` where the exact numbers you see may be different, you're all set to move on to step #5.

If for some reason you don't have Python or if you want to get the current version, you can now do this easily with Homebrew! Anytime you use Homebrew, you will start your command with `brew` followed by the Homebrew command you want to use. To install the latest version of python 2, simply type:
	
```bash
brew install python
brew install python3
```
	
If you'd rather install the latest version of python 3, replace `python` with `python3`.
	

*Install pip*
	
There are a few package managers that are specific to Python, and pip is the preferred one. The name pip stands for "pip installs packages". pip has one dependeny--distribute, but Homebrew doesn't know how to install either one. Luckily, both distribute and pip can be easily installed with python scripts that are available on the web. We can use `curl`, just like we did to get Homebrew. 

```bash
curl -O http://python-distribute.org/distribute_setup.py
python distribute_setup.py
curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
python get-pip.py
```
	    
This time we are getting and executing each script in two commands, where we did it all in one command before. Remember that you can look up what `-O` does with `$ man curl`, if you're curious.

It's possible that you will run into a permission issue here. Every file on your computer stores information about who can access and modify it. The get-pip.py script is going to try to write files to some of your system directories and it's possible that your user account doesn't have the right permissions. You can get around that though. If you get an error for one of these Python commands about permissions, type `sudo` before the rest of the command. Sudo stands for "superuser do". The superuser does have permission to modify system files and when you say sudo, you are acting as the superuser. You will need the admin password to do this.

```bash
brew install qt qt5 || brew install pyqt  # qt4 is deprecated
pip install labelme
```


Usage
-----

**Annotation**

Run `labelme --help` for detail.

```bash
labelme  # Open GUI
labelme static/apc2016_obj3.jpg  # Specify file
labelme static/apc2016_obj3.jpg -O static/apc2016_obj3.json  # Close window after the save
```

The annotations are saved as a [JSON](http://www.json.org/) file. The
file includes the image itself.

**Visualization**

To view the json file quickly, you can use utility script:

```bash
labelme_draw_json static/apc2016_obj3.json
```

**Convert to Dataset**

To convert the json to set of image and label, you can run following:


```bash
labelme_json_to_dataset static/apc2016_obj3.json
```


Sample
------

- [Original Image](https://github.com/wkentaro/labelme/blob/master/static/apc2016_obj3.jpg)
- [Screenshot](https://github.com/wkentaro/labelme/blob/master/static/apc2016_obj3_screenshot.jpg)
- [Generated Json File](https://github.com/wkentaro/labelme/blob/master/static/apc2016_obj3.json)
- [Visualized Json File](https://github.com/wkentaro/labelme/blob/master/static/apc2016_obj3_draw_json.jpg)


Screencast
----------



Acknowledgement
---------------

This repo is the fork of [mpitid/pylabelme](https://github.com/mpitid/pylabelme), whose development is currently stopped.
