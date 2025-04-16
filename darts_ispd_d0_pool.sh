###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-11-20 10:52:04
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ispd_dataset/
device_id=0
task_name=sig2s_ispd1_d0_npvb

for i in $(seq 45 -1 22); do
    layout_path=$layout_folder"ispd_c"$i".png"
    echo "Starting $task_name mo on device $device_id for: problems [darts] mask "$i
    echo $layout_path
    $python src/bilevel.py task_name=$task_name problems=pool module.device_id=$device_id mask.layout_path=$layout_path mask.target_path=$layout_path
done
