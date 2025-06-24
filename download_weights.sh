#!/bin/bash

#base_url="https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6WEIGHT.pt"
reopt_base_url="https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6"

mkdir weights

for weight in s n
do
    model_name="yolov6"$weight"_v2_reopt.pt"
    echo "Downloading $model_name"
    if [ "$weight" = "s" ]; then
        url=$reopt_base_url$weight"_v2_reopt.pt"
    else
        url=$reopt_base_url$weight"_v2_repopt.pt"
    fi
    wget -q $url -O weights/$model_name
    echo "Finished downloading $model_name"
done
