###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://gjchen.me
 # @Date: 2023-10-14 11:00:14
 # @LastEditTime: 2023-10-15 08:33:55
 # @Contact: cgjcuhk@gmail.com
 # @Description:
###
python=/home/local/eda13/gc29434/miniconda3/envs/smo/bin/python
layout_folder=/home/local/eda13/gc29434/phd/projects/SMO-dev-dac24/data/ibm_opc_test/mask
# source_list="conventional annular quasar dipole"
source_list="annular"

for type in $source_list; do
    for i in $(seq 4 4 16); do
        mask_sigmoid_steepness=$i
        for j in $(seq 60 15 90); do
            resist_sigmoid_steepness=$j
            for k in $(seq 7000 1000 9000); do
                weight_pvb=$k
                for l in $(seq 1 10); do
                    layout_path=$layout_folder"t"$l"_0_mask.png"
                    echo "Starting mo for: "$mask_sigmoid_steepness $resist_sigmoid_steepness $weight_pvb
                    $python src/moabbe.py task_name="mo_mss"$mask_sigmoid_steepness"rss"$resist_sigmoid_steepness"pvb"$weight_pvb mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type model.mask_sigmoid_steepness=$mask_sigmoid_steepness model.resist_sigmoid_steepness=$resist_sigmoid_steepness model.weight_pvb=$weight_pvb
                done
            done
        done
    done
done
