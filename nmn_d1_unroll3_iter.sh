###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-11-10 18:05:00
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
# layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ibm_opc_test/mask/
# layout_path=$layout_folder"t"$i"_0_mask.png"
device_id=1
problems_type=nmn
unroll_steps=3
nmn_iters="20 15 10 5 3 1"

for i in $nmn_iters; do
    echo "Starting bismo for: problems ["$problems_type"] mask1 unroll_steps="$unroll_steps nmn_iters=$i
    echo src/bilevel.py module.device_id=$device_id problems=$problems_type problems.nmn_iters=$i problems.unroll_steps=$unroll_steps
    $python src/bilevel.py module.device_id=$device_id problems=$problems_type problems.nmn_iters=$i problems.unroll_steps=$unroll_steps
done
