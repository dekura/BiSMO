#!/bin/bash
###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-05-01 01:25:56
 # @LastEditTime: 2023-10-04 16:28:31
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###


python=/home/gjchen21/miniconda3/envs/smo/bin/python

# source_list="conventional annular quasar dipole"
source_list="annular"
layout_folder="/home/local/eda13/gc29434/phd/projects/SMO-torch/data"

for type in $source_list; do
    for i in $(seq 1 2); do
        layout_path=$layout_folder"/ibm_opc_test/mask/t"$i"_0_mask.png"
        echo "Starting smo for: "$layout_path $type
        $python src/smo.py task_name="smo_"$type mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type
    done
done
