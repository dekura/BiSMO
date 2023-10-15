python=/home/hehq/anaconda3/envs/smo/bin/python

# source_list="conventional annular quasar dipole"
source_list="annular"

for type in $source_list; do
    for i in $(seq 4 4 16); do
        source_sigmoid_steepness=$i
        for j in $(seq 60 15 90); do
            resist_sigmoid_steepness=$j
            for k in $(seq 7000 1000 9000); do
                weight_pvb=$k
                for l in $(seq 1 10); do
                    layout_path="/home/hehq/project/smo2.0/data/img2013/M1_test"$l".png"
                    echo "Starting so for: "$source_sigmoid_steepness $resist_sigmoid_steepness $weight_pvb
                    $python src/soabbe.py task_name="so_sss"$source_sigmoid_steepness"rss"$resist_sigmoid_steepness"pvb"$weight_pvb mask.layout_path=$layout_path mask.target_path=$layout_path source.source_type=$type model.source_sigmoid_steepness=$source_sigmoid_steepness model.resist_sigmoid_steepness=$resist_sigmoid_steepness model.weight_pvb=$weight_pvb
                done
            done
        done
    done
done
