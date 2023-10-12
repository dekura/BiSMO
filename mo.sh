python=/home/hehq/anaconda3/envs/smo/bin/python

# source_list="conventional annular quasar dipole"
source_list="annular"

for type in $source_list; do
    for i in $(seq 1 10); do
        layout_path="/home/hehq/project/smo2.0/data/img2013/M1_test"$i".png"
        for j in $(seq 4 4 16); do
            mask_sigmoid_steepness=$j
            for k in $(seq 0.125 0.05 0.375); do
                resist_sigmoid_tr=$k
                for l in $(seq 30 15 75); do
                    resist_sigmoid_steepness=$l
                    echo "Starting mo for: "$layout_path $type $mask_sigmoid_steepness $resist_sigmoid_steepness
                    $python src/moabbe.py -m task_name="mo_mss"$mask_sigmoid_steepness"_rss"$resist_sigmoid_steepness"_rst"$resist_sigmoid_tr mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type model.mask_sigmoid_steepness=$mask_sigmoid_steepness model.resist_sigmoid_steepness=$resist_sigmoid_steepness model.resist_sigmoid_tr=$resist_sigmoid_tr
                done
            done
        done
    done
done
