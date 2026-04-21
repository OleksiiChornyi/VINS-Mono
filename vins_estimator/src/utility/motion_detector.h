#pragma once

#include <deque>
#include <cmath>
#include <eigen3/Eigen/Dense>

// Rolling-window motion classifier used by the hover-aware fork.
//
// Answers three related questions about the last window_sec seconds:
//
//   1. isStationary()   — accelerometer ≈ gravity, gyro ≈ 0, flow ≈ 0.
//                         Gates ZUPT, static-init bias priming, and failure
//                         softening.
//
//   2. isRotationOnly(focal_px) — gyroscope shows significant angular rate
//                         but the optical flow is explainable by rotation
//                         alone (no residual translational flow). True for
//                         a handheld device being panned/tilted in place,
//                         or a drone yawing at a stationary hover point.
//                         Used to apply a position-only ZUPT when VIO's
//                         translation estimate would otherwise drift due
//                         to extrinsic calibration errors leaking rotation
//                         into translation.
//
//   3. hasStationaryReference() / stationaryReferenceAcc() — the most
//                         recent meanAcc observed while isStationary() was
//                         true. Survives the transition out of stationary,
//                         so a post-init gravity-direction sanity check
//                         can run AGAINST this reference even when init
//                         completes during takeoff motion.
class MotionDetector
{
public:
    MotionDetector()
        : first_still_t_(-1.0),
          is_still_(false),
          last_t_(-1.0),
          last_flow_t_(-1.0),
          have_still_ref_(false),
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
        have_still_ref_ = false;
        still_ref_acc_.setZero();
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

    double stationaryDuration() const
    {
        if (first_still_t_ < 0 || last_t_ < 0) return 0.0;
        return last_t_ - first_still_t_;
    }

    Eigen::Vector3d meanGyr() const
    {
        Eigen::Vector3d s = Eigen::Vector3d::Zero();
        if (imu_buf_.empty()) return s;
        for (const auto &e : imu_buf_) s += e.gyr;
        return s / double(imu_buf_.size());
    }

    Eigen::Vector3d meanAcc() const
    {
        Eigen::Vector3d s = Eigen::Vector3d::Zero();
        if (imu_buf_.empty()) return s;
        for (const auto &e : imu_buf_) s += e.acc;
        return s / double(imu_buf_.size());
    }

    // Mean angular-rate magnitude (not magnitude of mean — different when
    // gyro has DC bias). Used by rotation-only classifier.
    double meanGyrMagnitude() const
    {
        if (imu_buf_.empty()) return 0.0;
        double s = 0.0;
        for (const auto &e : imu_buf_) s += e.gyr.norm();
        return s / double(imu_buf_.size());
    }

    // Peak |acc - g| seen over the window. Non-zero during motor spin-up
    // and sharp thrust transients; used by init to recognise "we are in
    // a takeoff event" and apply stricter acceptance criteria.
    double peakAccDeviation() const
    {
        double m = 0.0;
        for (const auto &e : imu_buf_)
        {
            double d = std::fabs(e.acc.norm() - gravity_norm_);
            if (d > m) m = d;
        }
        return m;
    }

    double meanFlow() const
    {
        if (flow_buf_.empty()) return 0.0;
        double s = 0.0;
        for (const auto &e : flow_buf_) s += e.second;
        return s / double(flow_buf_.size());
    }

    int imuCount() const { return (int)imu_buf_.size(); }
    int flowCount() const { return (int)flow_buf_.size(); }

    // Last meanAcc sampled while the detector was confirmed stationary.
    // Valid for the entire session once the device has been held still
    // at any point (typical: pre-arm on the ground).
    bool hasStationaryReference() const { return have_still_ref_; }
    const Eigen::Vector3d& stationaryReferenceAcc() const { return still_ref_acc_; }

    // Pure rotation classifier.
    //
    // Intuition: for a rigid rotation at angular rate ω with no translation,
    // the optical flow of an off-centre feature is approximately |ω|·f (in
    // pixels/sec), where f is the focal length. Translational flow is
    // (|v|/depth)·f; for handheld distances (~1–3 m) even small translation
    // produces flow comparable to rotational flow, so the ratio discriminates
    // rotation-only from general motion.
    //
    //   flow_measured < ratio · |ω|·f + baseline
    // is the acceptance rule. ratio ~ 1.3 is forgiving to feature depth
    // spread; baseline absorbs tracker noise at low ω.
    //
    // Gated on |ω| > gyr_min to avoid misclassifying slow translation as
    // rotation-dominated (where division/ratio thresholds become unstable).
    bool isRotationOnly(double focal_px, double gyr_min_rad_s,
                        double flow_ratio, double flow_baseline_px) const
    {
        if (imu_buf_.size() < 4) return false;
        if (flow_buf_.empty())   return false;  // need visual evidence

        double mean_gyr_mag = meanGyrMagnitude();
        if (mean_gyr_mag < gyr_min_rad_s) return false;

        double predicted_flow = mean_gyr_mag * focal_px;
        double measured_flow  = meanFlow();
        return measured_flow < flow_ratio * predicted_flow + flow_baseline_px;
    }

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

            // Cache meanAcc the moment we confirm stationary — this is the
            // most accurate gravity-in-body-frame measurement we will ever
            // get, and it stays valid as a reference after the drone starts
            // moving.
            if (is_still_)
            {
                still_ref_acc_ = meanAcc();
                have_still_ref_ = true;
            }
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

    // Sticky reference — last meanAcc while confirmed stationary.
    bool   have_still_ref_;
    Eigen::Vector3d still_ref_acc_;

    double acc_thr_;
    double gyr_thr_;
    double flow_thr_;
    double window_sec_;
    double hold_sec_;
    double gravity_norm_;
};
