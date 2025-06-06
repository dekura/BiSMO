###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-10-27 16:17:49
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ibm_opc_test/mask/
device_id=0
problems_type=cg

for i in $(seq 10 -1 1); do
    layout_path=$layout_folder"t"$i"_0_mask.png"
    echo "Starting bismo for: problems ["$problems_type"] mask "$i
    $python src/bilevel.py module.device_id=$device_id problems=$problems_type mask.layout_path=$layout_path mask.target_path=$layout_path
done
