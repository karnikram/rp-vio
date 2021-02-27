#! /bin/bash

dataset=$1

rm -r $1/results/viode
mkdir -p $1/results/viode
cd $1/results/viode
sed -i "s@~@$HOME@g" ~/test_ws/src/rp-vio/config/viode_config.yaml

echo -e "######viode-citynight######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator viode_citynight.launch bagfile_path:=$dataset/3_high.bag
    cp ~/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/test_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/high_groundtruth.txt est_traj_$i.txt --align --save_plot est_traj_$i.pdf |& tee -a report.txt
done
