###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-14 11:00:14
 # @LastEditTime: 2023-10-14 11:26:05
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
source_list="annular"

for type in $source_list; do
    for i in $(seq 1 1); do
        layout_path="/home/hehq/project/smo2.0/data/img2013/M1_test"$i".png"
        echo "Starting mo for: "$layout_path $type $mask_sigmoid_steepness $resist_sigmoid_steepness $resist_sigmoid_tr $mask_sigmoid_steepness $resist_sigmoid_steepness $resist_sigmoid_tr $mask_sigmoid_tr $weight_pvb
        $python src/moabbe.py mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type
    done
done
