python=/home/hehq/anaconda3/envs/smo/bin/python

# source_list="conventional annular quasar dipole"
source_list="annular"

for type in $source_list; do
    for i in $(seq 1 10); do
        layout_path="/home/hehq/project/smo2.0/data/img2013/M1_test"$i".png"
        for j in $(seq 4 4 12); do
            mask_sigmoid_steepness=$j
            for k in $(seq 0.075 0.05 0.225); do
                resist_sigmoid_tr=$k
                for l in $(seq 60 15 90); do
                    resist_sigmoid_steepness=$l
                    for m in $(seq 0.175 0.05 0.275); do
                        mask_sigmoid_tr=$m
                        for n in $(seq 7000 1000 9000); do
                            weight_pvb=$n
                            echo "Starting mo for: "$layout_path $type $mask_sigmoid_steepness $resist_sigmoid_steepness $resist_sigmoid_tr $mask_sigmoid_steepness $resist_sigmoid_steepness $resist_sigmoid_tr $mask_sigmoid_tr $weight_pvb
                            $python src/moabbe.py task_name="masked_mss"$mask_sigmoid_steepness"mst"$mask_sigmoid_tr"_rss"$resist_sigmoid_steepness"_rst"$resist_sigmoid_tr"pvb"$weight_pvb mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type model.mask_sigmoid_steepness=$mask_sigmoid_steepness model.resist_sigmoid_steepness=$resist_sigmoid_steepness model.resist_sigmoid_tr=$resist_sigmoid_tr model.mask_sigmoid_tr=$mask_sigmoid_tr model.weight_pvb=$weight_pvb
                        done
                    done
                done
            done
        done
    done
done
