#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <condition_variable>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "imu_processing.hpp"
#include "ivox3d/ivox3d.h"
#include "options.h"
#include "pointcloud_preprocess.h"

#include "visual_sparse_map.h"

#include "vins_estimator/estimator.h"
#include "vins_estimator/visualization.h"

#include <serial/serial.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace faster_lio {

struct SendOdometryBySerialTCP
{
    double Time_;
    common::V3D Pos_;
    common::V3D Vel_;
    Eigen::Quaterniond Q_;
    unsigned char s_buffer[52];

    std::string out_pose_file_ = "/home/casia/yhp_code/faster_livo_ws/out_pose_file.txt";
    std::ofstream fs_;

    int clientSocket_;
    struct sockaddr_in serverAddr_;

    SendOdometryBySerialTCP();
    void sendBySerial();
};

class LaserMapping {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;
#endif

    LaserMapping();
    ~LaserMapping() {
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        LOG(INFO) << "laser mapping deconstruct";
    }

    /// init with ros
    bool InitROS(ros::NodeHandle &nh);

    /// init without ros
    bool InitWithoutROS(const std::string &config_yaml);

    void Run();

    // callbacks of lidar and imu
    void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);

    // USE_VIO
    cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr& img_msg);
    void IMAGECallBack(const sensor_msgs::ImageConstPtr& msg);
    std::shared_ptr<Estimator> vins_estimator_ = nullptr;
    std::shared_ptr<FeatureTrackerLetnet> tracker_letnet_ = nullptr;
    std::shared_ptr<VisualSubMap> visual_sparse_map_ = nullptr;


    // sync lidar with imu
    bool SyncPackages();

    /// interface of mtk, customized obseravtion model
    void ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

    ////////////////////////////// debug save / show ////////////////////////////////////////////////////////////////
    void PublishPath(const ros::Publisher pub_path);
    void PublishOdometry(const ros::Publisher &pub_odom_aft_mapped);
    void PublishFrameWorld();
    void PublishFrameBody(const ros::Publisher &pub_laser_cloud_body);
    void PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world);
    void Savetrajectory(const std::string &traj_file);

    void PublishPathGrav(const ros::Publisher pub_path_grav);
    // void PublishPathFuse(const ros::Publisher pub_path_fuse);

    void PublishOdometryGrav(const ros::Publisher &pub_odom_grav_sys);
    // void PublishOdometryFuse(const ros::Publisher &pub_odom_fuse);

    void PublishOdomPathFused(const ros::Publisher &pub_odom_fused, const ros::Publisher &pub_path_fused);

    void Finish();

    // FeatureTrackerLetnet TrackerLetnet_;

private:
    template <typename T>
    void SetPosestamp(T &out);

    template <typename T>
    void SetPosestampGrav(T &out);

    template <typename T>
    void SetPosestampFuse(T &out);

    void PointBodyToWorld(PointType const *pi, PointType *const po);
    void PointBodyToWorld(const common::V3F &pi, PointType *const po);
    void PointBodyToWorldGravity(PointType const *pi, PointType *const po);
    void PointBodyToWorldGravity(const common::V3F &pi, PointType *const po);
    
    void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);

    void MapIncremental();

    void SubAndPubToROS(ros::NodeHandle &nh);

    bool LoadParams(ros::NodeHandle &nh);
    bool LoadParamsFromYAML(const std::string &yaml);

    void PrintState(const state_ikfom &s);

    common::V3D PosCur();
    common::M3D RotCur();

private:
    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process

    /// local map related
    float det_range_ = 300.0f;
    double cube_len_ = 0;
    double filter_size_map_min_ = 0;
    bool localmap_initialized_ = false;

    /// params
    std::vector<double> extrinT_{3, 0.0};  // lidar-imu translation 雷达在IMU坐标系下的坐标，这里写成下角标应该是T_il，有点误导人的hhh
    std::vector<double> extrinR_{9, 0.0};  // lidar-imu rotation
    std::string map_file_path_;

    /// point clouds data
    CloudPtr scan_undistort_{new PointCloudType()};   // scan after undistortion    去畸变的点云 body系
    CloudPtr scan_down_body_{new PointCloudType()};   // downsampled scan in body   去畸变后降采样的点云 body系
    CloudPtr scan_down_world_{new PointCloudType()};  // downsampled scan in world  去畸变后降采样的点云 LIO的world系
    CloudPtr scan_undistort_gravity_{new PointCloudType()};     // 

    std::vector<PointVector> nearest_points_;         // nearest points of current scan
    common::VV4F corr_pts_;                           // inlier pts, xyz normal
    common::VV4F corr_norm_;                          // inlier plane norms
    pcl::VoxelGrid<PointType> voxel_scan_;            // voxel filter for current scan
    std::vector<float> residuals_;                    // point-to-plane residuals
    std::vector<bool> point_selected_surf_;           // selected points
    common::VV4F plane_coef_;                         // plane coeffs

    common::V3D lidar_T_wrt_IMU_;
    common::M3D lidar_R_wrt_IMU_;

    /// ros pub and sub stuffs
    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Publisher pub_laser_cloud_world_;
    ros::Publisher pub_laser_cloud_body_;
    ros::Publisher pub_laser_cloud_effect_world_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;
    std::string tf_imu_frame_;
    std::string tf_world_frame_;

    ros::Publisher pub_path_grav_;
    ros::Publisher pub_odom_aft_mapped_grav_;

    ros::Publisher pub_path_fused_;
    ros::Publisher pub_odom_fused_;

    std::mutex mtx_buffer_;
    std::deque<double> time_buffer_;
    std::deque<PointCloudType::Ptr> lidar_buffer_;
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;

    nav_msgs::Odometry odom_aft_mapped_;
    nav_msgs::Odometry odom_aft_mapped_grav_;
    nav_msgs::Odometry odom_fused_;

    ///////////////////////// USE_VIO ///////////////////////////////
    ros::Subscriber sub_img_;
    std::deque<cv::Mat> img_buffer_;
    std::deque<double>  img_time_buffer_;
    double last_timestamp_img_ = 0.0;
    bool img_en_ = true;
    int img_w_;
    int img_h_;
    std::string vins_config_;
    bool initialize_vio_gravity_ = false;
    std::vector<double> extrinT_cl_{3, 0.0};  // camera->lidar translation
    std::vector<double> extrinR_cl_{9, 0.0};  // camera->lidar rotation
    common::V3D camera_T_wrt_lidar_;
    common::M3D camera_R_wrt_lidar_;

    /// options
    bool time_sync_en_ = false;
    double timediff_lidar_wrt_imu_ = 0.0;
    double last_timestamp_lidar_ = 0;
    double lidar_end_time_ = 0;
    double last_timestamp_imu_ = -1.0;
    double first_lidar_time_ = 0.0;
    bool lidar_pushed_ = false;

    /// statistics and flags ///
    int scan_count_ = 0;
    int publish_count_ = 0;
    bool flg_first_scan_ = true;
    bool flg_EKF_inited_ = false;
    int pcd_index_ = 0;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;
    bool timediff_set_flg_ = false;
    int effect_feat_num_ = 0, frame_num_ = 0;

    bool flg_LIO_optimized_ = false;

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    common::MeasureGroup measures_;                    // sync IMU and lidar scan
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf
    state_ikfom state_point_;                          // ekf current state
    vect3 pos_lidar_;                                  // lidar position after eskf update
    common::V3D euler_cur_ = common::V3D::Zero();      // rotation in euler angles
    bool extrinsic_est_en_ = true;

    /////////////////////////  debug show / save /////////////////////////////////////////////////////////
    bool run_in_offline_ = false;
    bool path_pub_en_ = true;
    bool scan_pub_en_ = false;
    bool dense_pub_en_ = false;
    bool scan_body_pub_en_ = false;
    bool scan_effect_pub_en_ = false;
    bool pcd_save_en_ = false;
    bool runtime_pos_log_ = true;
    int pcd_save_interval_ = -1;
    bool path_save_en_ = false;
    std::string dataset_;

    PointCloudType::Ptr pcl_wait_save_{new PointCloudType()};  // debug save

    nav_msgs::Path path_;
    geometry_msgs::PoseStamped msg_body_pose_;

    nav_msgs::Path path_grav_;  // 转换到重力系下
    geometry_msgs::PoseStamped msg_body_pose_grav_;

    nav_msgs::Path path_fused_;  // 与VIO融合后的
    geometry_msgs::PoseStamped msg_body_pose_fused_;

    std::deque<nav_msgs::Odometry> lio_odom_queue_; 

    common::M3D R_grav_imu_;

    SendOdometryBySerialTCP send_odom_serial_;
};



}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_H