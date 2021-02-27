#! /bin/bash

dataset=$1

rm -r $dataset/results/advio
mkdir -p $dataset/results/advio
cd $dataset/results/advio
sed -i "s@~@$HOME@g" ~/catkin_ws/src/rp-vio/config/advio_12_config.yaml

echo -e "######advio-12######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator advio_12.launch bagfile_path:=$dataset/12-mask.bag |& tee -a run_log.txt
    cp ~/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/catkin_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/groundtruth.txt est_traj_$i.txt --align --save_plot est_traj_$i.pdf |& tee -a report.txt
done
