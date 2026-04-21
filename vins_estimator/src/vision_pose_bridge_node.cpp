#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <mavros_msgs/State.h>
#include <mutex>
#include <cmath>

/*
 * VisionPoseBridge — forwards VINS-Mono odometry to ArduPilot via
 * /mavros/vision_pose/pose with sanity checking and recovery, and mirrors
 * the sanitized pose on /vins_bridge/pose for downstream consumers
 * (web UI, loggers) that want the same authoritative stream MAVROS sees.
 *
 * States:
 *   PRE_INIT  →  Publishes (0,0,0) + identity orientation at ~rate Hz.
 *                Allows arm in PosHold.
 *                Transition: VINS odometry arrives (passes sanity) → ACTIVE
 *
 *   ACTIVE    →  Forwards VINS position + orientation.
 *                Transitions:
 *                  - VINS data stale > odom_timeout → DROPOUT
 *                  - Acceleration sanity check fails → DROPOUT
 *
 *   DROPOUT   →  Stops publishing. EKF dead-reckons.
 *                Transitions:
 *                  - VINS resumes (passes sanity) → ACTIVE
 *                  - Drone is disarmed → PRE_INIT
 *
 * Sanity check: acceleration-based.
 *   Computes implied velocity from consecutive positions, then
 *   computes acceleration from consecutive velocities.
 *   If acceleration > max_accel → position jumped → DROPOUT.
 *
 * Published topics:
 *   /mavros/vision_pose/pose  — what ArduPilot's EKF3 consumes
 *   /vins_bridge/pose         — mirror for external consumers
 *
 * Parameters (ROS):
 *   ~max_accel     — max plausible acceleration [m/s^2] (default: 25.0)
 *   ~odom_timeout  — seconds without data before DROPOUT (default: 1.0)
 *   ~rate          — publish rate Hz (default: 20.0)
 */

class VisionPoseBridge
{
    enum class State { PRE_INIT, ACTIVE, DROPOUT };

public:
    VisionPoseBridge()
        : state_(State::PRE_INIT)
        , is_armed_(false)
        , last_odom_time_(0)
        , have_prev_pos_(false)
        , have_prev_vel_(false)
        , sanity_violations_(0)
    {
        ros::NodeHandle pnh("~");
        pnh.param("max_accel", max_accel_, 25.0);
        pnh.param("odom_timeout", odom_timeout_, 1.0);
        double rate;
        pnh.param("rate", rate, 10.0);

        pub_mavros_ = nh_.advertise<geometry_msgs::PoseStamped>(
            "/mavros/vision_pose/pose", 10);
        pub_mirror_ = nh_.advertise<geometry_msgs::PoseStamped>(
            "/vins_bridge/pose", 10);
        sub_odom_ = nh_.subscribe(
            "/vins_estimator/odometry", 10,
            &VisionPoseBridge::odomCallback, this);
        sub_state_ = nh_.subscribe(
            "/mavros/state", 5,
            &VisionPoseBridge::stateCallback, this);

        timer_ = nh_.createTimer(
            ros::Duration(1.0 / rate),
            &VisionPoseBridge::timerCallback, this);

        ROS_INFO("vision_pose_bridge: started (%.0f Hz)", rate);
        ROS_INFO("  max_accel=%.1f m/s^2, odom_timeout=%.1fs",
                 max_accel_, odom_timeout_);
    }

private:
    // ── Acceleration-based sanity check ───────────────────────────
    bool checkSanity(const geometry_msgs::Point& pos, double stamp)
    {
        if (!have_prev_pos_)
        {
            prev_pos_ = pos;
            prev_stamp_ = stamp;
            prev_dt_ = 0;
            have_prev_pos_ = true;
            have_prev_vel_ = false;
            sanity_violations_ = 0;
            return true;
        }

        double dt = stamp - prev_stamp_;
        if (dt < 0.001)
            return true;

        double vx = (pos.x - prev_pos_.x) / dt;
        double vy = (pos.y - prev_pos_.y) / dt;
        double vz = (pos.z - prev_pos_.z) / dt;

        prev_pos_ = pos;
        prev_stamp_ = stamp;

        if (!have_prev_vel_)
        {
            prev_vx_ = vx; prev_vy_ = vy; prev_vz_ = vz;
            prev_dt_ = dt;
            have_prev_vel_ = true;
            sanity_violations_ = 0;
            return true;
        }

        double dt_accel = (prev_dt_ + dt) * 0.5;
        if (dt_accel < 0.001) dt_accel = dt;

        double ax = (vx - prev_vx_) / dt_accel;
        double ay = (vy - prev_vy_) / dt_accel;
        double az = (vz - prev_vz_) / dt_accel;
        double accel = std::sqrt(ax*ax + ay*ay + az*az);

        prev_vx_ = vx; prev_vy_ = vy; prev_vz_ = vz;
        prev_dt_ = dt;

        if (accel > max_accel_)
        {
            sanity_violations_++;
            ROS_WARN("vision_pose_bridge: accel=%.1f m/s^2 "
                     "(max=%.1f), violations=%d",
                     accel, max_accel_, sanity_violations_);
            if (sanity_violations_ >= 2)
                return false;
            return true;
        }

        sanity_violations_ = 0;
        return true;
    }

    void resetSanity()
    {
        have_prev_pos_ = false;
        have_prev_vel_ = false;
        prev_dt_ = 0;
        sanity_violations_ = 0;
    }

    // ── Callbacks ─────────────────────────────────────────────────
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        double now = ros::Time::now().toSec();

        if (!checkSanity(msg->pose.pose.position, now))
        {
            if (state_ == State::ACTIVE)
            {
                state_ = State::DROPOUT;
                resetSanity();
                ROS_WARN("vision_pose_bridge: sanity fail -> DROPOUT");
            }
            return;
        }

        last_odom_time_ = now;
        last_pose_.header = msg->header;
        last_pose_.header.frame_id = "odom";
        last_pose_.pose.position = msg->pose.pose.position;
        last_pose_.pose.orientation = msg->pose.pose.orientation;

        if (state_ != State::ACTIVE)
        {
            State prev = state_;
            state_ = State::ACTIVE;
            ROS_INFO("vision_pose_bridge: -> ACTIVE%s",
                     prev == State::PRE_INIT ? " (initialized)" : " (recovered)");
        }
    }

    void stateCallback(const mavros_msgs::State::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        is_armed_ = msg->armed;
    }

    // ── Timer ─────────────────────────────────────────────────────
    void timerCallback(const ros::TimerEvent&)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        double now = ros::Time::now().toSec();

        if (state_ == State::ACTIVE)
        {
            if (now - last_odom_time_ > odom_timeout_)
            {
                state_ = State::DROPOUT;
                resetSanity();
                ROS_WARN("vision_pose_bridge: stale -> DROPOUT");
                return;
            }
        }
        else if (state_ == State::DROPOUT)
        {
            if (!is_armed_)
            {
                state_ = State::PRE_INIT;
                resetSanity();
                ROS_INFO("vision_pose_bridge: disarmed -> PRE_INIT");
            }
            else
                return;
        }

        geometry_msgs::PoseStamped out;
        out.header.stamp = ros::Time::now();
        out.header.frame_id = "odom";

        if (state_ == State::ACTIVE)
        {
            out.pose = last_pose_.pose;
        }
        else  // PRE_INIT
        {
            out.pose.position.x = 0;
            out.pose.position.y = 0;
            out.pose.position.z = 0;
            out.pose.orientation.x = 0;
            out.pose.orientation.y = 0;
            out.pose.orientation.z = 0;
            out.pose.orientation.w = 1;
        }

        pub_mavros_.publish(out);
        pub_mirror_.publish(out);
    }

    ros::NodeHandle nh_;
    ros::Publisher pub_mavros_;
    ros::Publisher pub_mirror_;
    ros::Subscriber sub_odom_;
    ros::Subscriber sub_state_;
    ros::Timer timer_;
    std::mutex mtx_;

    State state_;
    bool is_armed_;
    double last_odom_time_;
    double max_accel_;
    double odom_timeout_;

    geometry_msgs::Point prev_pos_;
    double prev_stamp_;
    bool have_prev_pos_;
    double prev_vx_, prev_vy_, prev_vz_;
    double prev_dt_;
    bool have_prev_vel_;
    int sanity_violations_;

    geometry_msgs::PoseStamped last_pose_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vision_pose_bridge");
    VisionPoseBridge bridge;
    ros::spin();
    return 0;
}
