#pragma once

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace Eigen;

#include <ros/console.h>

class MotionEstimator
{
  public:

    bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);
    bool solveRelativeHRT(const vector<pair<Vector3d, Vector3d>> &corres, const Matrix3d &R_imu, const Matrix4d &TrIC, Matrix3d &R, Vector3d &T, Vector3d &n);
    void decomposeH(const cv::Mat &H, const cv::Mat &K, const Matrix3d &R_imu, const  Matrix4d &TrIC, const Vector3d &mean_l, Matrix4d &est_Tr, Vector3d &est_n);

  private:
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};


