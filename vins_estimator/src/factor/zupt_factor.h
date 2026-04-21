#pragma once

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

// Zero-Velocity Update (ZUPT) factors used when the platform is detected to
// be stationary. They inject pseudo-measurements that the velocity is zero
// and that consecutive positions are equal, which stops IMU-integrated drift
// during prolonged hover — the regime where VIO is geometrically starved.
//
// Weights are configured from YAML (see ZUPT_VEL_WEIGHT / ZUPT_POS_WEIGHT).
// Intuition: weight ≈ 1 / sigma, where sigma is the expected noise of the
// pseudo-measurement. For a drone hovering on attitude-hold, residual drift
// is typically ~1 cm/s in velocity and < 1 mm between frames in position,
// which maps to weights of ~100 and ~1000 respectively.

// Residual (3): velocity in world frame at a single window index.
// Parameter block: para_SpeedBias[j] (9: [V(3) Ba(3) Bg(3)]).
class ZUPTVelocityFactor : public ceres::SizedCostFunction<3, 9>
{
  public:
    explicit ZUPTVelocityFactor(double weight)
        : sqrt_info_(weight * Eigen::Matrix3d::Identity()) {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        Eigen::Vector3d Vj(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt_info_ * Vj;

        if (jacobians && jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            J.block<3, 3>(0, 0) = sqrt_info_;
        }
        return true;
    }

  private:
    Eigen::Matrix3d sqrt_info_;
};

// Residual (3): position delta Pj - Pi in world frame.
// Parameter blocks: para_Pose[i], para_Pose[j] (7 each: [P(3) Q(4)]).
class ZUPTPositionFactor : public ceres::SizedCostFunction<3, 7, 7>
{
  public:
    explicit ZUPTPositionFactor(double weight)
        : sqrt_info_(weight * Eigen::Matrix3d::Identity()) {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt_info_ * (Pj - Pi);

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Ji(jacobians[0]);
                Ji.setZero();
                Ji.block<3, 3>(0, 0) = -sqrt_info_;
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Jj(jacobians[1]);
                Jj.setZero();
                Jj.block<3, 3>(0, 0) = sqrt_info_;
            }
        }
        return true;
    }

  private:
    Eigen::Matrix3d sqrt_info_;
};
