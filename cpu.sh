#!/bin/bash
###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-05-01 01:25:56
 # @LastEditTime: 2023-10-06 16:01:10
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###


python=/home/local/eda13/gc29434/miniconda3/envs/cpusmo/bin/python

# source_list="conventional annular quasar dipole"
source_list="annular"
layout_folder="/home/local/eda13/gc29434/phd/projects/SMO-torch/data"
thread_list="188 256"

for type in $source_list; do
    for nth in $thread_list; do
        for i in $(seq 1 1); do
            layout_path=$layout_folder"/ibm_opc_test/mask/t"$i"_0_mask.png"
	    echo -e "\n\n=========================================================\n"
            echo "Using "$nth"threads, starting smo for: "$layout_path $type
            $python src/smo.py experiment=smallsource num_threads=$nth trainer=cpu task_name="smo_"$type mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type
	    echo -e "\n\n=========================================================\n"
        done
    done
done
