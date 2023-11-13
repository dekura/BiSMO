###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-11-13 09:36:37
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ibm_large_test/
device_id=1
task_name=sig2s_large

for i in $(seq 20 -1 11); do
    layout_path=$layout_folder"t"$i"_0_mask.png"
    echo "Starting $task_name mo for: problems [darts] mask "$i
    $python src/bilevel.py task_name=$task_name problems=pool module.device_id=$device_id mask.layout_path=$layout_path mask.target_path=$layout_path
done
