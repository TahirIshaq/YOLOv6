#!/bin/bash
# Downloads yolov6s and yolov6n weights from 0.4.o release

base_url="https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6WEIGHT.pt"

for weight in s n
do
    echo "Downloading yolov6$weight"
    url="${base_url/WEIGHT/"$weight"}"
    wget -q $url
    echo "Finished downloading yolov6$weight"
done