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

// Hover-aware extensions. All values are optional in the YAML — the reader
// supplies a sensible default when a key is missing. "hardcoded" thresholds
// that the user asked to remove (0.5 m/s runaway, 25° rotation disagreement,
// 8° gravity check, 3 m/s init cap, 150 frames reset, ZUPT weights) are now
// derived from IMU noise characteristics (ACC_N / GYR_N) and the measured
// image rate — they live as scaling coefficients rather than absolute values.
extern int    ENABLE_ZUPT;
extern double STATIC_ACC_THR;
extern double STATIC_GYR_THR;
extern double STATIC_FLOW_THR;
extern double STATIC_WINDOW_SEC;
extern int    STATIC_CONFIRM_CYCLES;       // hysteresis, evaluate()-cycles
extern int    STATIC_INIT_BIAS_PRIMING;
extern int    SOFTEN_FAILURE_ON_HOVER;
extern int    ENABLE_ROTATION_ZUPT;
extern double ROTATION_ZUPT_GYR_MIN;
extern double ROTATION_ZUPT_FLOW_RATIO;
extern double ROTATION_ZUPT_FLOW_BASELINE;

// Adaptive scale factors (not absolute values) — user knobs with physical
// meaning, default to 1.0:
//   * GRAVITY_CHECK_SIGMA  scales the per-platform noise floor when turning
//     the init gravity-disagreement threshold into degrees.
//   * ROTATION_DISAGREE_SIGMA does the same for the IMU-vs-visual init
//     rotation check.
//   * INIT_MAX_VEL_COEF scales the IMU-derived plausibility cap for |V|
//     after init (cap = coef · g · sum_dt — i.e. the max |V| attainable by
//     integrating gravity over the window, a hard physical limit).
//   * ZUPT_WEIGHT_SCALE biases the adaptive ZUPT weights toward stronger
//     anchoring (>1) or softer hold (<1).
//   * INIT_PARALLAX_PX is in normalized-FOCAL pixel units (VINS convention
//     with FOCAL_LENGTH = 460); unchanged by image resolution.
extern double GRAVITY_CHECK_SIGMA;
extern double ROTATION_DISAGREE_SIGMA;
extern double INIT_MAX_VEL_COEF;
extern double ZUPT_WEIGHT_SCALE;
extern double INIT_PARALLAX_PX;

// Runaway detection — multiplier on IMU-consistent |V|. Default 4.0 means
// "|V| is four-sigma above what IMU says we could be moving". No hardcoded
// absolute velocity involved.
extern double RUNAWAY_IMU_SIGMA;

// Safety-reset thresholds. `IDLE_RESET_SECONDS` is "how many seconds of
// idle all_image_frame growth before we bail" — converted to frames using
// the observed image rate (so image rate changes are handled correctly).
extern double IDLE_RESET_SECONDS;
extern int    IDLE_RESET_MAX_ATTEMPTS;

// Vision-pose bridge speed/accel limits. These come from the vehicle's
// physics envelope, not from per-flight tuning — they can live in YAML:
//   * VEHICLE_MAX_ACCEL  [m/s²]  — maximum plausible acceleration
//   * VEHICLE_MAX_SPEED  [m/s]   — maximum plausible ground speed
// The bridge uses the accel limit as a spike filter (warn but accept) and
// the speed limit as the hard kick-out criterion (position truly diverged).
extern double VEHICLE_MAX_ACCEL;
extern double VEHICLE_MAX_SPEED;

// Optional legacy YAML overrides. ≤0 → use the adaptive / derived value.
// Exposed so existing per-platform YAMLs keep working without edits.
extern double ZUPT_VEL_WEIGHT_OVERRIDE;
extern double ZUPT_POS_WEIGHT_OVERRIDE;
extern double ROTATION_ZUPT_POS_WEIGHT_OVERRIDE;
extern double INIT_MAX_VELOCITY_OVERRIDE;
extern double GRAVITY_CHECK_ANGLE_DEG_OVERRIDE;
extern double RUNAWAY_V_ABS_OVERRIDE;


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
