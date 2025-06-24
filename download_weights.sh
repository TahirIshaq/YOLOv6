#!/bin/bash

base_url="https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6WEIGHT.pt"
mkdir weights

for weight in s n
do
    echo "Downloading yolov6$weight"
    url="${base_url/WEIGHT/"$weight"}"
    wget -q $url -O weights/yolov6$weight.pt
    echo "Finished downloading yolov6$weight.pt"
done
