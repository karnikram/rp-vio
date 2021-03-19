## RP-VIO: Robust Plane-based Visual-Inertial Odometry for Dynamic Environments

<p align="center">
<a href="https://user-images.githubusercontent.com/12653355/111314569-88dfb000-8687-11eb-87c8-212f7ad13489.png"><img src="https://user-images.githubusercontent.com/12653355/111314569-88dfb000-8687-11eb-87c8-212f7ad13489.png" width="700"/></a>
</p>
RP-VIO is a monocular visual-inertial odometry (VIO) system that uses only planar features and their induced homographies, during both initialization and sliding-window estimation, for increased robustness and accuracy in dynamic environments.

**Pre-print**: https://arxiv.org/abs/2103.10400

**Introductory video**<br><br>
[![Intro video](https://user-images.githubusercontent.com/12653355/111311553-5e402800-8684-11eb-85bd-0db5b7494772.png)](https://youtu.be/2GMoUJEDO0U "RP-VIO: Intro Video")

## Setup
Our evaluation setup is a 6-core Intel Core i5-8400 CPU with 8GB RAM and a 1 TB HDD, running Ubuntu 18.04.1. We recommend using a more powerful setup, especially for heavy datasets like ADVIO or OpenLORIS.

### Pre-requisites
[ROS Melodic](http://wiki.ros.org/melodic) (OpenCV 3.2.0, Eigen 3.3.4-4)<br>
[Ceres Solver 1.14.0](https://github.com/ceres-solver/ceres-solver/releases)<br>
[EVO](https://github.com/MichaelGrupp/evo)

The versions indicated are the versions used in our evaluation setup, and we do not guarantee our code to run on newer versions like ROS Noetic (OpenCV 4.2).

### Build
Run the following commands in your terminal to clone our project and build,

```
    cd ~/catkin_ws/src
    git clone https://github.com/karnikram/rp-vio.git
    cd ../
    catkin_make -j4
    source ~/catkin_ws/devel/setup.bash
```

## Evaluation
We provide evaluation scripts to run RP-VIO on the [RPVIO-Sim](https://github.com/karnikram/rp-vio#rpvio-sim-dataset-1) dataset, and select sequences from the [OpenLORIS-Scene]((https://lifelong-robotic-vision.github.io/dataset/scene.html)), [ADVIO](https://github.com/AaltoVision/ADVIO), and [VIODE](https://github.com/kminoda/VIODE) datasets. The output errors from your evaluation might not be exactly the same as reported in our paper, but should be similar.

### RPVIO-Sim Dataset
Download the [dataset](https://github.com/karnikram/rp-vio#rpvio-sim-dataset-1) files to a parent folder, and then run the following commands to launch our evaluation script. The script runs rp-vio on each of the six sequences once and computes the ATE error statistics.

```
    cd ~/catkin_ws/src/rp-vio/scripts/
    ./run_rpvio_sim.sh <PATH-TO-DATASET-FOLDER>
```

To run the multiple planes version (RPVIO-Multi), checkout the corresponding branch by running `git checkout rpvio-multi`, and re-run the above script.

### Real-world sequences
We evaluate on two real-world sequences: the market1-1 sequence from the OpenLORIS-Scene dataset and the metro station sequence (12) from the ADVIO dataset. Both of these sequences along with their segmented plane masks are available as bagfiles for download [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/robotics_iiit_ac_in/EozZ6vJP5UZFmZhA-9w0bBcBvTXpszD42fPx3x3ZlKvD6A?e=FtzFRz). After downloading and extracting the files run the following commands for evaluation,

```
    cd ~/catkin_ws/src/rp-vio/scripts/
    ./run_ol_market1.sh <PATH-TO-EXTRACTED-DATASET-FOLDER>
    ./run_advio_12.sh <PATH-TO-EXTRACTED-DATASET-FOLDER>
```

### Own data
To run RP-VIO on your own data, you need to provide synchronized monocular images, IMU readings, and plane masks on three separate ROS topics. The camera and IMU need to be properly calibrated and synchronized as there is no online calibration. A plane segmentation model to segment plane masks from images is provided [below](https://github.com/karnikram/rp-vio#plane-segmentation).

A semantic segmentation model can also be as long as the RGB labels of the (static) planar semantic classes are provided. As an example, we evaluate on a sequence from the VIODE dataset (provided [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/robotics_iiit_ac_in/EoxFVvuAxUdFsnXu0XJY0egBMFxB9D8XbNqe0jkUkRdjVg?e=G18fDo)) using semantic segmentation labels which are specified in the [config file](https://github.com/karnikram/rp-vio/blob/semantic-viode/config/viode_config.yaml). To run, 

```
    cd ~/catkin_ws/src/rp-vio/scripts
    git checkout semantic-viode
    ./run_viode_night.sh <PATH-TO-EXTRACTED-DATASET-FOLDER>
```

## Plane segmentation
We provide a pre-trained plane instance segmentation model, based on the [Plane-Recover](https://github.com/fuy34/planerecover) model. We retrained their model, with an added inter-plane constraint, on their SYNTHIA training data and two additional sequences (00,01) from the ScanNet dataset. The model was trained on a single Titan X (maxwell) GPU for about 700K iterations. We also provide the training script.

We run the model offline, after extracting and [processing](https://github.com/fuy34/planerecover#preparing-training-data) the input RGB images from their ROS bagfiles. To run the pre-trained model (requires CUDA 9.0),
```
cd plane_segmentation/
conda create --name plane_seg --file requirements.txt
conda activate plane_seg
python inference.py --dataset=<PATH_TO_DATASET> --output_dir=<PATH_TO_OUTPUT_DIRECTORY> --test_list=<TEST_DATA_LIST.txt FILE> --ckpt_file=<MODEL> --use_preprocessed=true 
```

We also use a dense CRF model (from [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf)) to further refine the output masks. To run,

```
python crf_inference.py <rgb_image_dir> <labels_dir> <output_dir>
```

We then write these outputs mask images back into the original bagfile on a separate topic for running RP-VIO.

## RPVIO-Sim Dataset
<figure>
<a href="https://user-images.githubusercontent.com/12653355/111727645-48538280-8891-11eb-90db-027f82087586.png"><img src="https://user-images.githubusercontent.com/12653355/111727645-48538280-8891-11eb-90db-027f82087586.png" width="400"/></a>
</figure>
<br>

For an effective evaluation of the capabilities of modern VINS systems, we generate a highly-dynamic visual-inertial dataset using [AirSim](https://github.com/microsoft/AirSim/) which contains dynamic characters present throughout the sequences (including initialization), and with sufficient IMU excitation. Dynamic characters are progressively added, keeping everything else fixed, starting from no characters in the `static` sequence to eight characters in the `C8` sequence. All the generated sequences (six) in rosbag format, along with their groundtruth files, have been made available via [Zenodo](https://zenodo.org/record/4603494#.YE4BzlMzZH4).

Each rosbag contains RGB images published on the `/image` topic at 20 Hz, imu measurements published on the`/imu` topic at ~1000 Hz (which we sub-sample to 200Hz for our evaluations), and plane-instance mask images published on the`/mask` topic at 20 Hz. The groundtruth trajectory is saved as a txt file in TUM format. The parameters for the camera and IMU used in our dataset are as follows,
<br>
<figure>
<a href="https://user-images.githubusercontent.com/12653355/111068192-c3ade080-84ed-11eb-82ba-486ee0cfa2a4.png"><img src="https://user-images.githubusercontent.com/12653355/111068192-c3ade080-84ed-11eb-82ba-486ee0cfa2a4.png" width="300"/></a>
</figure>

To quantify the dynamic nature of our generated sequences, we compute the percentage of dynamic pixels out of all the pixels present in every image. We report these values in the following table,
<figure>
<a href="https://user-images.githubusercontent.com/12653355/111068119-6f0a6580-84ed-11eb-86ee-12571e7c7476.png"><img src="https://user-images.githubusercontent.com/12653355/111068119-6f0a6580-84ed-11eb-86ee-12571e7c7476.png" width="400"/></a>
</figure>

### TO-DO
- [ ] Provide Unreal Engine environment
- [ ] Provide AirSim recording scripts

## Acknowledgement
Our code is built upon [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono). Its implementations of feature tracking, IMU preintegration, IMU state initialization, the reprojection factor, and marginalization are used as such. Our contributions include planar features tracking, planar homography based initialization, and the planar homography factor. All these changes (corresponding to a slightly older version) are available as a [git patch file](./rpvio.patch).

For our simulated dataset, we imported several high-quality assets from the [FlightGoggles](https://flightgoggles.mit.edu/) project into [Unreal Engine](unrealengine.com) before integrating it with AirSim. The dynamic characters were downloaded from [Mixamo](https://mixamo.com).
