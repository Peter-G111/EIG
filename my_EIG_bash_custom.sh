#!/bin/bash

### runs `EIG_Notebook.jl` with different parameters (customised in code in this file)
### $1 : total timeout (secs) for each oracle

# usage: EIG_Notebook.jl <total_timeout for each oracle int> <t_max_offset int> <use_orig_t_max bool> <grid_size> <trial_num>
# echo "running custom t_max run 0 offset with grid_size=$1 and trial_num=$2, and ult_timeout=$3"
# julia "EIG_Notebook.jl" $3 "0" "false" $1 $2


# run the original t_max experiments
echo "running original t_max runs"

for grid_size in $(seq 6 6);
do
    for trial_num in $(seq 8 10);
    do
        echo "running grid_size=$grid_size trial_num=$trial_num orig t_max with total_timeout=$1"
        
        # usage: EIG_Notebook.jl <total_timeout for each oracle int> <t_max_offset int> <use_orig_t_max bool> <grid_size> <trial_num>
        julia "EIG_Notebook.jl" $1 "0" "true" $grid_size $trial_num
    done
done