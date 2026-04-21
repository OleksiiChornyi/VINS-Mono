#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;

// Hover-aware extensions (see config/*.yaml for defaults).
extern int    ENABLE_ZUPT;
extern double ZUPT_VEL_WEIGHT;
extern double ZUPT_POS_WEIGHT;
extern double STATIC_ACC_THR;
extern double STATIC_GYR_THR;
extern double STATIC_FLOW_THR;
extern double STATIC_WINDOW_SEC;
extern double STATIC_HOLD_SEC;
extern int    STATIC_INIT_BIAS_PRIMING;
extern int    SOFTEN_FAILURE_ON_HOVER;
extern double HOVER_MIN_PARALLAX_FACTOR;
extern double INIT_MAX_VELOCITY;
extern double GRAVITY_CHECK_ANGLE_DEG;
extern int    ENABLE_ROTATION_ZUPT;
extern double ROTATION_ZUPT_GYR_MIN;
extern double ROTATION_ZUPT_FLOW_RATIO;
extern double ROTATION_ZUPT_FLOW_BASELINE;
extern double ROTATION_ZUPT_POS_WEIGHT;


void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
