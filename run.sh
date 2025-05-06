#!/bin/bash

rm -rf build
mkdir build
cd build

cmake ..
make

# ./app yolo11n.ncnn.param yolo11n.ncnn.bin
./app model.ncnn.param model.ncnn.bin