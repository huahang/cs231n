#!/bin/sh

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install numpy
pip install scipy
pip install matplotlib
pip install ipykernel
pip install future
