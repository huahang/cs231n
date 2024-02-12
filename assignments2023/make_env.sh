#!/bin/sh

conda create -n cs231n python=3.8
conda activate cs231n
conda install \
  -c pytorch \
  -c nvidia \
  -c conda-forge \
  pytorch-cuda \
  pytorch \
  torchvision \
  jupyterlab \
