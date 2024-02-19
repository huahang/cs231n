#!/bin/sh

conda create -n cs231n python=3.8
conda activate cs231n
conda install \
  -c pytorch \
  -c nvidia \
  -c conda-forge \
  pytorch \
  torchvision \
  jupyterlab \
  matplotlib \
  h5py \
  imageio \
  scipy \
  numpy \
  opencv \
  pytorch-model-summary \
  hdf5 \

OS=$(uname)

if [ "$OS" = "Linux" ]; then
    echo "This is Linux"
    conda install -c pytorch -c nvidia -c conda-forge pytorch-cuda
elif [ "$OS" = "Darwin" ]; then
    echo "This is Mac"
else
    echo "Unknown OS"
fi
