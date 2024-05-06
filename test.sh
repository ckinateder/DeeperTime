#! /bin/bash

folder_name=${1:-ETTm2}
exp_name=${2:-96M}
stor_name=$exp_name
# if folder_name is ILI, then if exp_name is 24M, set it to 96M, if exp_name is 36M, set it to 192M
if [ $folder_name == "ILI" ]; then
    if [ $exp_name == "24M" ]; then
        stor_name="96M"
    elif [ $exp_name == "36M" ]; then
        stor_name="192M"
    elif [ $exp_name == "48M" ]; then
        stor_name="336M"
    elif [ $exp_name == "60M" ]; then
        stor_name="720M"
    fi
fi


echo Running $folder_name/$exp_name ...
echo Removing old experiment...
rm -rf storage/experiments/$folder_name/$stor_name
echo Building experiment...
make build config=experiments/configs/$folder_name/$exp_name.gin
echo Running experiment @ storage/experiments/$folder_name/$stor_name ...
python -m experiments.forecast --config_path=storage/experiments/$folder_name/$stor_name/repeat\=0/config.gin run
echo Done.