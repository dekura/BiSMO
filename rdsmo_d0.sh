###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-11-03 13:22:03
 # @LastEditTime: 2023-11-03 16:43:09
 # @Contact: cgjcuhk@gmail.com
 # @Description:
### 



python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ibm_opc_test/mask/
device_id=0
alter_epoch=10
task_name=rd_ibm10

for i in $(seq 10 -1 1); do
    layout_path=$layout_folder"t"$i"_0_mask.png"
    echo "Starting rdsmo for: $task_name mask "$i
    echo  python src/rdsmo.py trainer.devices="[$device_id]" model.alter_epoch=$alter_epoch task_name=$task_name mask.layout_path=$layout_path
    $python src/rdsmo.py trainer.devices="[$device_id]" model.alter_epoch=$alter_epoch task_name=$task_name mask.layout_path=$layout_path mask.target_path=$layout_path
done
