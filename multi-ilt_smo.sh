#!/bin/bash
###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-05-01 01:25:56
 # @LastEditTime: 2023-05-19 21:36:32
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###


python=/home/gjchen21/miniconda3/envs/smo/bin/python

# source_list="conventional annular quasar dipole"
source_list="annular"
dataset_name="multi-ilt-center-test"


for type in $source_list; do
    for i in $(seq 1 10); do
        layout_path="/home/gjchen21/projects/smo/SMO-torch/data/multi-ilt/center-test/mask/mask_exact_"$i".png"
        target_path="/home/gjchen21/projects/smo/SMO-torch/data/multi-ilt/center-test/target/mask_exact_"$i".png"
        echo "Starting smo for: "$layout_path $type
        $python src/smo.py task_name="milt_"$type mask.layout_path=$layout_path mask.target_path=$target_path source.source_type=$type mask.dataset_name=$dataset_name
    done
done
