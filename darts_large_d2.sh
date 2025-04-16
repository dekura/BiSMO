###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-11-12 22:48:17
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ibm_large_test/
device_id=2

for i in $(seq 20 -1 11); do
    layout_path=$layout_folder"t"$i"_0_mask.png"
    echo "Starting mo for: problems [darts] mask "$i
    $python src/bilevel.py module.device_id=$device_id mask.layout_path=$layout_path mask.target_path=$layout_path
done
