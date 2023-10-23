###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-10-23 17:39:30
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python

num_list="4 15 20"

for i in $num_list; do
    $python src/bilevel.py module.device_id=0 problems=cg problems.cg_iters=$i
done