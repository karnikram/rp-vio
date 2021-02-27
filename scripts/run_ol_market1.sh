#! /bin/bash

dataset=$1

rm -r $dataset/results/ol
mkdir -p $dataset/results/ol
cd $dataset/results/ol
sed -i "s@~@$HOME@g" ~/catkin_ws/src/rp-vio/config/ol_market1_config.yaml

echo -e "######ol-market1######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator ol_market1.launch bagfile_path:=$dataset/market1-mask.bag
    cp ~/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/catkin_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/groundtruth.txt est_traj_$i.txt --align --save_plot est_traj_$i.pdf |& tee -a report.txt
done
