#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW, COL;
double TD, TR;

int    ENABLE_ZUPT;
double STATIC_ACC_THR;
double STATIC_GYR_THR;
double STATIC_FLOW_THR;
double STATIC_WINDOW_SEC;
int    STATIC_CONFIRM_CYCLES;
int    STATIC_INIT_BIAS_PRIMING;
int    SOFTEN_FAILURE_ON_HOVER;
int    ENABLE_ROTATION_ZUPT;
double ROTATION_ZUPT_GYR_MIN;
double ROTATION_ZUPT_FLOW_RATIO;
double ROTATION_ZUPT_FLOW_BASELINE;

double GRAVITY_CHECK_SIGMA;
double ROTATION_DISAGREE_SIGMA;
double INIT_MAX_VEL_COEF;
double ZUPT_WEIGHT_SCALE;
double INIT_PARALLAX_PX;

double RUNAWAY_IMU_SIGMA;

double IDLE_RESET_SECONDS;
int    IDLE_RESET_MAX_ATTEMPTS;

double VEHICLE_MAX_ACCEL;
double VEHICLE_MAX_SPEED;

// Optional overrides kept for backward-compatibility with the previous
// fork's YAML. If > 0, they replace the adaptive computation; ≤ 0 means
// "use adaptive" (recommended). Not documented as first-class knobs.
double ZUPT_VEL_WEIGHT_OVERRIDE;
double ZUPT_POS_WEIGHT_OVERRIDE;
double ROTATION_ZUPT_POS_WEIGHT_OVERRIDE;
double INIT_MAX_VELOCITY_OVERRIDE;
double GRAVITY_CHECK_ANGLE_DEG_OVERRIDE;
double RUNAWAY_V_ABS_OVERRIDE;

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

    fsSettings["imu_topic"] >> IMU_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;

    // create folder if not exists
    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());

    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
        
    } 

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }

    // Hover-aware extensions. Every field is optional and falls back to a
    // default tuned for a small multirotor hovering at altitude — the
    // scenario where vanilla VINS-Mono drifts the most.
    auto readOr = [&](const char *key, double def) -> double {
        cv::FileNode n = fsSettings[key];
        if (n.empty() || (!n.isReal() && !n.isInt())) return def;
        return static_cast<double>(n);
    };
    auto readOrInt = [&](const char *key, int def) -> int {
        cv::FileNode n = fsSettings[key];
        if (n.empty() || !n.isInt()) return def;
        return static_cast<int>(n);
    };

    ENABLE_ZUPT              = readOrInt("enable_zupt",              1);
    STATIC_ACC_THR           = readOr   ("static_acc_thr",           0.5);
    STATIC_GYR_THR           = readOr   ("static_gyr_thr",           0.05);
    STATIC_FLOW_THR          = readOr   ("static_flow_thr",          2.0);
    STATIC_WINDOW_SEC        = readOr   ("static_window_sec",        0.5);
    // Hysteresis is in evaluate()-cycles, not seconds. At ~200 Hz IMU rate
    // 3 cycles ≈ 15 ms confirmation delay — fast enough for a drone yet
    // robust against single-sample spikes.
    STATIC_CONFIRM_CYCLES    = readOrInt("static_confirm_cycles",    3);
    STATIC_INIT_BIAS_PRIMING = readOrInt("static_init_bias_priming", 1);
    SOFTEN_FAILURE_ON_HOVER  = readOrInt("soften_failure_on_hover",  1);
    ENABLE_ROTATION_ZUPT     = readOrInt("enable_rotation_zupt",     1);
    ROTATION_ZUPT_GYR_MIN    = readOr   ("rotation_zupt_gyr_min",    0.3);
    ROTATION_ZUPT_FLOW_RATIO = readOr   ("rotation_zupt_flow_ratio", 1.3);
    ROTATION_ZUPT_FLOW_BASELINE = readOr("rotation_zupt_flow_baseline", 30.0);

    // Adaptive-threshold scale factors. Defaults = 1 mean "use the derived
    // value as-is"; tune >1 to loosen, <1 to tighten.
    GRAVITY_CHECK_SIGMA      = readOr   ("gravity_check_sigma",      6.0);
    ROTATION_DISAGREE_SIGMA  = readOr   ("rotation_disagree_sigma",  8.0);
    INIT_MAX_VEL_COEF        = readOr   ("init_max_vel_coef",        1.5);
    ZUPT_WEIGHT_SCALE        = readOr   ("zupt_weight_scale",        1.0);
    // Parallax threshold in normalized-FOCAL pixel units (FOCAL_LENGTH=460
    // by VINS convention). Same value works across camera resolutions
    // because features are normalized before parallax is computed.
    INIT_PARALLAX_PX         = readOr   ("init_parallax_px",         18.0);

    RUNAWAY_IMU_SIGMA        = readOr   ("runaway_imu_sigma",        6.0);

    IDLE_RESET_SECONDS       = readOr   ("idle_reset_seconds",       15.0);
    IDLE_RESET_MAX_ATTEMPTS  = readOrInt("idle_reset_max_attempts",  3);

    // Platform-level physics limits. These are *vehicle* constants, not
    // per-flight tuning. Defaults are for a small multirotor; larger
    // airframes need higher. The bridge uses accel as a "spike-filter
    // warning" and speed as the "hard divergence" cutoff, so setting
    // them generously is safer than tightly.
    VEHICLE_MAX_ACCEL        = readOr   ("vehicle_max_accel",        30.0);
    VEHICLE_MAX_SPEED        = readOr   ("vehicle_max_speed",        20.0);

    // Legacy overrides (≤0 → adaptive).
    ZUPT_VEL_WEIGHT_OVERRIDE        = readOr("zupt_vel_weight",         -1.0);
    ZUPT_POS_WEIGHT_OVERRIDE        = readOr("zupt_pos_weight",         -1.0);
    ROTATION_ZUPT_POS_WEIGHT_OVERRIDE = readOr("rotation_zupt_pos_weight", -1.0);
    INIT_MAX_VELOCITY_OVERRIDE      = readOr("init_max_velocity",       -1.0);
    GRAVITY_CHECK_ANGLE_DEG_OVERRIDE = readOr("gravity_check_angle_deg", -1.0);
    RUNAWAY_V_ABS_OVERRIDE          = readOr("runaway_v_abs",           -1.0);

    ROS_INFO("hover-aware: zupt=%d acc_thr=%.3f gyr_thr=%.3f flow_thr=%.2f window=%.2fs cycles=%d",
             ENABLE_ZUPT, STATIC_ACC_THR, STATIC_GYR_THR, STATIC_FLOW_THR,
             STATIC_WINDOW_SEC, STATIC_CONFIRM_CYCLES);
    ROS_INFO("hover-aware: adaptive sigmas g=%.1f rot=%.1f vmax_coef=%.1f zupt_scale=%.2f parallax_px=%.1f",
             GRAVITY_CHECK_SIGMA, ROTATION_DISAGREE_SIGMA, INIT_MAX_VEL_COEF,
             ZUPT_WEIGHT_SCALE, INIT_PARALLAX_PX);
    ROS_INFO("hover-aware: idle_reset=%.1fs (max %d), runaway_imu_sigma=%.1f, vehicle max a=%.1f v=%.1f",
             IDLE_RESET_SECONDS, IDLE_RESET_MAX_ATTEMPTS, RUNAWAY_IMU_SIGMA,
             VEHICLE_MAX_ACCEL, VEHICLE_MAX_SPEED);
    ROS_INFO("hover-aware: rot_zupt=%d (gyr_min=%.2f, ratio=%.2f, baseline=%.0fpx)",
             ENABLE_ROTATION_ZUPT, ROTATION_ZUPT_GYR_MIN,
             ROTATION_ZUPT_FLOW_RATIO, ROTATION_ZUPT_FLOW_BASELINE);

    fsSettings.release();
}
