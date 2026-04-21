#pragma once

#include <deque>
#include <eigen3/Eigen/Dense>

// Rolling-window stationarity detector fusing IMU variance with optical-flow
// magnitude. Used by the estimator to trigger Zero-Velocity Updates (ZUPT),
// prime bias/gravity estimates during init, and soften failure detection
// while the drone is stationary at altitude (hover scenario).
//
// Decision rule: "stationary" = for at least `hold_sec` continuous seconds,
//   - accelerometer magnitude deviates from gravity by < acc_thr
//   - gyro magnitude stays below gyr_thr
//   - mean optical flow over the window stays below flow_thr (px/sec)
// If no flow samples have arrived yet, the IMU-only criteria apply.
class MotionDetector
{
public:
    MotionDetector()
        : first_still_t_(-1.0),
          is_still_(false),
          last_t_(-1.0),
          last_flow_t_(-1.0),
          acc_thr_(0.5),
          gyr_thr_(0.05),
          flow_thr_(2.0),
          window_sec_(0.5),
          hold_sec_(0.5),
          gravity_norm_(9.81)
    {}

    void configure(double acc_thr, double gyr_thr, double flow_thr,
                   double window_sec, double hold_sec, double gravity_norm)
    {
        acc_thr_     = acc_thr;
        gyr_thr_     = gyr_thr;
        flow_thr_    = flow_thr;
        window_sec_  = window_sec;
        hold_sec_    = hold_sec;
        gravity_norm_ = gravity_norm;
    }

    void reset()
    {
        imu_buf_.clear();
        flow_buf_.clear();
        first_still_t_ = -1.0;
        is_still_      = false;
        last_t_        = -1.0;
        last_flow_t_   = -1.0;
    }

    void pushIMU(double t, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        imu_buf_.push_back({t, acc, gyr});
        trimWindow(imu_buf_, t);
        last_t_ = t;
        evaluate();
    }

    void pushFlow(double t, double flow_px_per_sec)
    {
        flow_buf_.push_back({t, flow_px_per_sec});
        trimFlowWindow(flow_buf_, t);
        last_flow_t_ = t;
    }

    bool isStationary() const { return is_still_; }

    // Duration (seconds) the system has been continuously stationary, 0 when
    // moving. Used to gate transitions that should only trigger after holding
    // still for a while (e.g. ZUPT weight ramp, static init).
    double stationaryDuration() const
    {
        if (first_still_t_ < 0 || last_t_ < 0) return 0.0;
        return last_t_ - first_still_t_;
    }

    // Mean gyro over current window. Good estimator for Bg when stationary.
    Eigen::Vector3d meanGyr() const
    {
        Eigen::Vector3d s = Eigen::Vector3d::Zero();
        if (imu_buf_.empty()) return s;
        for (const auto &e : imu_buf_) s += e.gyr;
        return s / double(imu_buf_.size());
    }

    // Mean acc over current window. When stationary this equals R_wb^T * g,
    // so it gives the gravity direction in the IMU frame.
    Eigen::Vector3d meanAcc() const
    {
        Eigen::Vector3d s = Eigen::Vector3d::Zero();
        if (imu_buf_.empty()) return s;
        for (const auto &e : imu_buf_) s += e.acc;
        return s / double(imu_buf_.size());
    }

    int imuCount() const { return (int)imu_buf_.size(); }

private:
    struct ImuSample { double t; Eigen::Vector3d acc; Eigen::Vector3d gyr; };

    void trimWindow(std::deque<ImuSample> &buf, double now)
    {
        while (!buf.empty() && now - buf.front().t > window_sec_)
            buf.pop_front();
    }

    void trimFlowWindow(std::deque<std::pair<double,double>> &buf, double now)
    {
        while (!buf.empty() && now - buf.front().first > window_sec_)
            buf.pop_front();
    }

    void evaluate()
    {
        if (imu_buf_.size() < 4) { is_still_ = false; first_still_t_ = -1.0; return; }

        double acc_dev_max = 0.0;
        double gyr_max     = 0.0;
        for (const auto &e : imu_buf_)
        {
            double acc_dev = std::fabs(e.acc.norm() - gravity_norm_);
            if (acc_dev > acc_dev_max) acc_dev_max = acc_dev;
            double g = e.gyr.norm();
            if (g > gyr_max) gyr_max = g;
        }

        bool imu_still = (acc_dev_max < acc_thr_) && (gyr_max < gyr_thr_);

        bool flow_still = true;
        if (!flow_buf_.empty())
        {
            double flow_sum = 0.0;
            for (const auto &e : flow_buf_) flow_sum += e.second;
            double flow_avg = flow_sum / double(flow_buf_.size());
            flow_still = (flow_avg < flow_thr_);
        }

        bool now_still = imu_still && flow_still;

        if (now_still)
        {
            if (first_still_t_ < 0) first_still_t_ = last_t_;
            is_still_ = (last_t_ - first_still_t_) >= hold_sec_;
        }
        else
        {
            first_still_t_ = -1.0;
            is_still_      = false;
        }
    }

    std::deque<ImuSample> imu_buf_;
    std::deque<std::pair<double,double>> flow_buf_;

    double first_still_t_;
    bool   is_still_;
    double last_t_;
    double last_flow_t_;

    double acc_thr_;
    double gyr_thr_;
    double flow_thr_;
    double window_sec_;
    double hold_sec_;
    double gravity_norm_;
};
