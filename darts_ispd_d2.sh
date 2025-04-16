###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-10-28 13:23:56
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ispd_dataset/
device_id=2

for i in $(seq 45 -1 1); do
    layout_path=$layout_folder"ispd_c"$i".png"
    echo "Starting mo on device $device_id for: problems [darts] mask "$i
    echo $layout_path
    $python src/bilevel.py module.device_id=$device_id mask.layout_path=$layout_path mask.target_path=$layout_path
done
