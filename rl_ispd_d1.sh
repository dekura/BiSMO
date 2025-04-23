###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-23 17:32:01
 # @LastEditTime: 2023-10-28 13:27:20
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-DAC24-codes/data/ispd_dataset/
device_id=1
problems_type=rl

for i in $(seq 45 -1 1); do
    layout_path=$layout_folder"ispd_c"$i".png"
    echo "Starting mo on device $device_id for: problems [$problems_type] mask $i"
    echo $layout_path
    $python src/bilevel.py module.device_id=$device_id problems=$problems_type mask.layout_path=$layout_path mask.target_path=$layout_path \
    problems.mo.config.policy_lr=0.0001 problems.mo.config.q_lr=0.0001 problems.mo.config.reward_lr=0.0003 \
    problems.mo.config.gamma=0.99 problems.mo.config.alpha=0.2 problems.mo.config.buffer_size=1000000 \
    problems.mo.config.batch_size=64 problems.mo.config.alternate=10000
done
