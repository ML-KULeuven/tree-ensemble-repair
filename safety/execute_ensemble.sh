#!/bin/bash

# Check number of arguments
if [ $# -eq 7 ]; then
    # one arguments provided: benchmark
    benchmark="$1"
    depth=$2
    num_trees=$3
    repair_method=$4
    data=$5
    lda=$6
    n_sample=$7
    og_flag=""
    if [ "$data" == "only_goal" ]; then
        og_flag="-og"
    fi
    pt_flag=""
    if [ "$repair_method" == "milp-l1" ]; then
        pt_flag="-pt True"
    fi
    python3 start_experiment.py -rv False -r $repair_method -exp repair -d $depth -t $num_trees -ls $lda -n $n_sample $og_flag $pt_flag "$benchmark"
elif [ $# -eq 8 ]; then
    # Three arguments provided: prefix, and benchmark
    prefix="$1"
    benchmark="$2"
    depth=$3
    num_trees=$4
    repair_method=$5
    data=$6
    lda=$7
    n_sample=$8
    og_flag=""
    if [ "$data" == "only_goal" ]; then
        og_flag="-og"
    fi
    pt_flag=""
    if [ "$repair_method" == "milp-l1" ]; then
        pt_flag="-pt True"
    fi
    python3 start_experiment.py -rv False -p "$prefix" -d $depth -t $num_trees -r $repair_method -exp repair -ls $lda -n $n_sample $og_flag $pt_flag "$benchmark"
else
    # Invalid number of arguments
    echo "Usage: $0 [prefix] benchmark"
    echo "  With 1 arguments: benchmark"
    echo "  With 2 arguments: prefix benchmark"
    exit 1
fi
