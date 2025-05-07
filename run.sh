#!/bin/bash

rm -rf build
mkdir build
cd build

cmake ..
make

./yolo_ncnn yolo12n_opt.param yolo12n_opt.bin