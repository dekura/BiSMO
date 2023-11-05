###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-14 11:00:14
 # @LastEditTime: 2023-11-05 14:38:31
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###

python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ispd_dataset/
device_id=2
task_name=milt_ispd
mask_acti=multi


for i in $(seq 45 -1 1); do
    layout_path=$layout_folder"ispd_c"$i".png"
    echo "Starting ispd mo for: $task_name mask "$i
    echo  python src/moabbe.py trainer.devices="[$device_id]" task_name=$task_name mask.layout_path=$layout_path
    $python src/moabbe.py trainer.devices="[$device_id]" model.mask_acti=$mask_acti task_name=$task_name mask.layout_path=$layout_path mask.target_path=$layout_path
done
