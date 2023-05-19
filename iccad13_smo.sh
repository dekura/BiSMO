#!/bin/bash
###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-05-01 01:25:56
 # @LastEditTime: 2023-05-19 21:02:31
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###


python=/home/gjchen21/miniconda3/envs/smo/bin/python

# source_list="conventional annular quasar dipole"
source_list="annular"

for type in $source_list; do
    for i in $(seq 1 10); do
        layout_path="/home/gjchen21/projects/smo/SMO-torch/data/ibm_opc_test/mask/t"$i"_0_mask.png"
        echo "Starting smo for: "$layout_path $type
        $python src/smo.py task_name="smo_"$type mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type
    done
done
