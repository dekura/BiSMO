###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-10-27 14:22:13
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
# layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ibm_opc_test/mask/
# layout_path=$layout_folder"t"$i"_0_mask.png"
device_id=0
problems_type=cg

unroll_steps="18 25"

for i in $unroll_steps; do
    echo "Starting mo for: problems ["$problems_type"] mask1 unroll_steps="$i
    $python src/bilevel.py module.device_id=$device_id problems=$problems_type problems.unroll_steps=$i
done
