###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-11-03 13:22:03
 # @LastEditTime: 2023-11-03 16:41:32
 # @Contact: cgjcuhk@gmail.com
 # @Description:
### 



python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ispd_dataset/
device_id=2
alter_epoch=10
task_name=rd_ispd

for i in $(seq 45 -1 1); do
    layout_path=$layout_folder"ispd_c"$i".png"
    echo "Starting rdsmo for: $task_name mask "$i
    echo  python src/rdsmo.py trainer.devices="[$device_id]" model.alter_epoch=$alter_epoch task_name=$task_name mask.layout_path=$layout_path
    $python src/rdsmo.py trainer.devices="[$device_id]" model.alter_epoch=$alter_epoch task_name=$task_name mask.layout_path=$layout_path mask.target_path=$layout_path
done
