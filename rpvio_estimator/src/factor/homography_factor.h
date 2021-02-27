#pragma once
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

struct HomographyFactor
{
    HomographyFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) : pts_i(_pts_i), pts_j(_pts_j) {}

    template <typename T>
        bool operator()(const T* const pose_i, const T* const pose_j, const T* const para_n, const T* const para_depth, const T* const ex_pose, 
        T* residuals) const
        {
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> pi(pose_i);
            Eigen::Quaternion<T> qi;
            qi.coeffs() << pose_i[3], pose_i[4], pose_i[5], pose_i[6];

            Eigen::Map<const Eigen::Matrix<T, 3, 1>> pj(pose_j);
            Eigen::Quaternion<T> qj;
            qj.coeffs() << pose_j[3], pose_j[4], pose_j[5], pose_j[6];

            Eigen::Map<const Eigen::Matrix<T, 3, 1>> tic(ex_pose);
            Eigen::Quaternion<T> qic;
            qic.coeffs() << ex_pose[3], ex_pose[4], ex_pose[5], ex_pose[6];

            Eigen::Matrix<T, 3, 1> n;
            n << para_n[0], para_n[1], para_n[2];

            // transform camera normal to imu normal
            Eigen::Matrix<T, 3, 1> n_imu_0 = qic*n;// + tic;

            // transform imu 0 normal to imu i normal
            Eigen::Matrix<T, 3, 1> n_imu_i = qi.inverse()*n_imu_0;// - qi.inverse()*pi;

            Eigen::Map<const Eigen::Matrix<T, 1, 1>> depth(para_depth);

            Eigen::Quaternion<T> qji = qj.inverse() * qi;
            Eigen::Matrix<T, 3, 1> tji = qj.inverse() * (pi - pj);
            Eigen::Matrix<T, 1, 1> di, di0;

            // convert camera depth to imu frame
            di0(0,0) = depth(0,0) + tic.dot(n_imu_0);
            // convert imu 0 depth to imu i depth
            di(0,0) = di0(0,0) - pi.dot(n_imu_0);

            Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_i.cast<T>() + tic;

            // homography mapping
            Eigen::Matrix<T, 3, 1> pts_imu_j = qji * pts_imu_i + (tji*(1.0/di(0,0)) * n_imu_i.transpose()) * pts_imu_i;

            Eigen::Matrix<T, 3, 1> pts_cam_j = qic.inverse() * (pts_imu_j - tic);

            pts_cam_j = (pts_cam_j / pts_cam_j[2]);
            residuals[0] = pts_cam_j[0] - pts_j[0];
            residuals[1] = pts_cam_j[1] - pts_j[1];

            return true;
        }

    static ceres::CostFunction* Create(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j)
    {
        return (new ceres::AutoDiffCostFunction<HomographyFactor, 2, 7, 7, 3, 1, 7>
                (new HomographyFactor(_pts_i, _pts_j)));
    }

    Eigen::Vector3d pts_i, pts_j;
};
