#include "parameters.h"

std::string IMAGE_TOPIC;
std::string MASK_TOPIC;
std::vector<cv::Vec3b> PLANES_BGR;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double H_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string RPVIO_FOLDER_PATH = readParam<std::string>(n, "rpvio_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["mask_topic"] >> MASK_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    
    cv::Mat planes_rgb;
    fsSettings["planesRGB"] >> planes_rgb;
    
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    H_THRESHOLD = fsSettings["H_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = RPVIO_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);
    
    for(int i = 0; i < planes_rgb.rows; i++)
    {
        cv::Vec3b plane_bgr(planes_rgb.at<uchar>(i, 2), planes_rgb.at<uchar>(i, 1), planes_rgb.at<uchar>(i, 0));
        PLANES_BGR.push_back(plane_bgr);
    }

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
