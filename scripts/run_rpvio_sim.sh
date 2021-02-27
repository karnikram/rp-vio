dataset=$1

rm -r $dataset/results/rpvio-sim/
mkdir -p $dataset/results/rpvio-sim/
cd $dataset/results/rpvio-sim/
sed -i "s@~@$HOME@g" ~/catkin_ws/src/rp-vio/config/rpvio_sim_config.yaml

########## static run #############

mkdir static
echo -e "######static######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator rpvio_sim.launch bagfile_path:=$dataset/static/static.bag
    cp /home/karnik/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/catkin_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv static/est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/static/groundtruth.txt static/est_traj_$i.txt --align --save_plot static/est_traj_$i.pdf |& tee -a report.txt
done

########## c1 run #############

mkdir c1
echo -e "######c1######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator rpvio_sim.launch bagfile_path:=$dataset/c1/c1.bag
    cp /home/karnik/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/catkin_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv c1/est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/c1/groundtruth.txt c1/est_traj_$i.txt --align --save_plot c1/est_traj_$i.pdf |& tee -a report.txt
done

########### c2 run #############

mkdir c2
echo -e "######c2######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator rpvio_sim.launch bagfile_path:=$dataset/c2/c2.bag
    cp /home/karnik/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/catkin_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv c2/est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/c2/groundtruth.txt c2/est_traj_$i.txt --align --save_plot c2/est_traj_$i.pdf |& tee -a report.txt
done

########### c4 run #############

mkdir c4
echo -e "######c4######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator rpvio_sim.launch bagfile_path:=$dataset/c4/c4.bag
    cp /home/karnik/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/catkin_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv c4/est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/c4/groundtruth.txt c4/est_traj_$i.txt --align --save_plot c4/est_traj_$i.pdf |& tee -a report.txt
done

############ c6 run #############

mkdir c6
echo -e "######c6######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator rpvio_sim.launch bagfile_path:=$dataset/c6/c6.bag
    cp /home/karnik/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/catkin_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv c6/est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/c6/groundtruth.txt c6/est_traj_$i.txt --align --save_plot c6/est_traj_$i.pdf |& tee -a report.txt
done

############ c8 run #############

mkdir c8
echo -e "######c8######\n" > report.txt

for((i = 1; i <=1; i++))
do
    roslaunch rpvio_estimator rpvio_sim.launch bagfile_path:=$dataset/c8/c8.bag
    cp /home/karnik/output/rpvio_result_no_loop.csv ./rpvio_est.csv
    python ~/catkin_ws/src/rp-vio/scripts/convert_vins_to_tum.py rpvio_est.csv c8/est_traj_$i.txt
    rm rpvio_est.csv
    evo_ape tum $dataset/c8/groundtruth.txt c8/est_traj_$i.txt --align --save_plot c8/est_traj_$i.pdf |& tee -a report.txt
done
