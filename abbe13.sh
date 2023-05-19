#!/bin/bash
###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-05-01 01:25:56
 # @LastEditTime: 2023-05-17 14:03:35
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###


python=/home/gjchen21/miniconda3/envs/smo/bin/python

source_list="conventional annular quasar dipole"

sig_out_array=($(seq 0.8 0.01 0.99))
sig_in_array=($(seq 0.55 0.01 0.8))

for sig_out in ${sig_out_array[@]}; do
    for sig_in in ${sig_in_array[@]}; do
        echo "sig_out: "$sig_out" sig_in: "$sig_in
        $python src/abbe.py source.sigma_out=$sig_out source.sigma_in=$sig_in
    done
done

# for type in $source_list; do
#     for i in $(seq 1 10); do
#         layout_path="/home/gjchen21/projects/smo/SMO-torch/data/ibm_opc_test/mask/"$i"_0_mask.png"
#         echo "Starting smo for: "$layout_path $type
#         $python src/smo.py task_name="smo_"$type mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type
#     done
# done
