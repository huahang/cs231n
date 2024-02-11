#!/bin/sh

conda install -c pytorch -c nvidia pytorch-cuda pytorch torchvision
pip install -U pip
pip install -r assignment3/requirements.txt
pip install -U numpy
pip install -U pillow
