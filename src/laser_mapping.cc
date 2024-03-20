#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <fstream>

#include "laser_mapping.h"
#include "utils.h"

namespace faster_lio {

bool LaserMapping::InitROS(ros::NodeHandle &nh) {
    LoadParams(nh);
    SubAndPubToROS(nh);

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // USE_VIO
    std::cout<<"vins_config: "<<std::string(std::string(ROOT_DIR) + "config/" + vins_config_)<<std::endl;
    registerPub(nh);
    vins_estimator_ = std::make_shared<Estimator>();
    tracker_letnet_ = std::make_shared<FeatureTrackerLetnet>();
    readParameters(std::string(std::string(ROOT_DIR) + "config/" + vins_config_));
    vins_estimator_->setParameter();    // 注意这里已经开启VINS的主线程等着数据进来，MULTIPLE_THREAD is 1，Estimator::processMeasurements 里面的while(1)
    // std::cout<<"LaserMapping: CAM_NAMES[0] "<<CAM_NAMES[0]<<std::endl;
    // TrackerLetnet_.readIntrinsicParameter(CAM_NAMES[0]);
    tracker_letnet_->readIntrinsicParameter(CAM_NAMES[0]);
    visual_sparse_map_ = std::make_shared<VisualSubMap>(CAM_NAMES[0],
                                                        std::string(std::string(ROOT_DIR) + "model/model.param"),
                                                        std::string(std::string(ROOT_DIR) + "model/model.bin"));
    visual_sparse_map_->setLidarCameraExtrinsic(camera_T_wrt_lidar_,camera_R_wrt_lidar_);
    visual_sparse_map_->setIMULidarExtrinsic(lidar_T_wrt_IMU_,lidar_R_wrt_IMU_);

    // esekf init 这是开多线程了吗
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        options::NUM_MAX_ITERATIONS, epsi.data());

    return true;
}

bool LaserMapping::InitWithoutROS(const std::string &config_yaml) {
    LOG(INFO) << "init laser mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        options::NUM_MAX_ITERATIONS, epsi.data());

    if (std::is_same<IVoxType, IVox<3, IVoxNodeType::PHC, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using phc ivox";
    } else if (std::is_same<IVoxType, IVox<3, IVoxNodeType::DEFAULT, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using default ivox";
    }

    return true;
}

bool LaserMapping::LoadParams(ros::NodeHandle &nh) {
    // get params from param server
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;

    nh.param<bool>("path_save_en", path_save_en_, true);
    nh.param<bool>("publish/path_publish_en", path_pub_en_, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en_, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en_, true);
    nh.param<bool>("publish/scan_effect_pub_en", scan_effect_pub_en_, false);
    nh.param<std::string>("publish/tf_imu_frame", tf_imu_frame_, "body");
    nh.param<std::string>("publish/tf_world_frame", tf_world_frame_, "camera_init");

    nh.param<int>("max_iteration", options::NUM_MAX_ITERATIONS, 4);
    nh.param<float>("esti_plane_threshold", options::ESTI_PLANE_THRESHOLD, 0.1);
    nh.param<std::string>("map_file_path", map_file_path_, "");
    nh.param<bool>("common/time_sync_en", time_sync_en_, false);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min_, 0.0);
    nh.param<double>("cube_side_length", cube_len_, 200);
    nh.param<float>("mapping/det_range", det_range_, 300.f);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
    nh.param<float>("preprocess/time_scale", preprocess_->TimeScale(), 1e-3);
    nh.param<int>("preprocess/lidar_type", lidar_type, 1);
    nh.param<int>("preprocess/scan_line", preprocess_->NumScans(), 16);
    nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
    nh.param<bool>("feature_extract_enable", preprocess_->FeatureEnabled(), false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_, true);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en_, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en_, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval_, -1);
    nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_, std::vector<double>());
    nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_, std::vector<double>());

    nh.param<float>("ivox_grid_resolution", ivox_options_.resolution_, 0.2);
    nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);

    // USE_VIO
    nh.param<std::string>("camera/vins_config_file", vins_config_, "vins_config");
    nh.param<std::vector<double>>("camera/Pcl", extrinT_cl_, std::vector<double>());
    nh.param<std::vector<double>>("camera/Rcl", extrinR_cl_, std::vector<double>());
    camera_T_wrt_lidar_ = common::VecFromArray<double>(extrinT_cl_);
    camera_R_wrt_lidar_ = common::MatFromArray<double>(extrinR_cl_);

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "camera_init";

    path_grav_.header.stamp = ros::Time::now();
    path_grav_.header.frame_id = "camera_init";

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU_ = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU_ = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU_, lidar_R_wrt_IMU_);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    // get params from yaml
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    auto yaml = YAML::LoadFile(yaml_file);
    try {
        path_pub_en_ = yaml["publish"]["path_publish_en"].as<bool>();
        scan_pub_en_ = yaml["publish"]["scan_publish_en"].as<bool>();
        dense_pub_en_ = yaml["publish"]["dense_publish_en"].as<bool>();
        scan_body_pub_en_ = yaml["publish"]["scan_bodyframe_pub_en"].as<bool>();
        scan_effect_pub_en_ = yaml["publish"]["scan_effect_pub_en"].as<bool>();
        tf_imu_frame_ = yaml["publish"]["tf_imu_frame"].as<std::string>("body");
        tf_world_frame_ = yaml["publish"]["tf_world_frame"].as<std::string>("camera_init");
        path_save_en_ = yaml["path_save_en"].as<bool>();

        options::NUM_MAX_ITERATIONS = yaml["max_iteration"].as<int>();
        options::ESTI_PLANE_THRESHOLD = yaml["esti_plane_threshold"].as<float>();
        time_sync_en_ = yaml["common"]["time_sync_en"].as<bool>();

        filter_size_surf_min = yaml["filter_size_surf"].as<float>();
        filter_size_map_min_ = yaml["filter_size_map"].as<float>();
        cube_len_ = yaml["cube_side_length"].as<int>();
        det_range_ = yaml["mapping"]["det_range"].as<float>();
        gyr_cov = yaml["mapping"]["gyr_cov"].as<float>();
        acc_cov = yaml["mapping"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["mapping"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["mapping"]["b_acc_cov"].as<float>();
        preprocess_->Blind() = yaml["preprocess"]["blind"].as<double>();
        preprocess_->TimeScale() = yaml["preprocess"]["time_scale"].as<double>();
        lidar_type = yaml["preprocess"]["lidar_type"].as<int>();
        preprocess_->NumScans() = yaml["preprocess"]["scan_line"].as<int>();
        preprocess_->PointFilterNum() = yaml["point_filter_num"].as<int>();
        preprocess_->FeatureEnabled() = yaml["feature_extract_enable"].as<bool>();
        extrinsic_est_en_ = yaml["mapping"]["extrinsic_est_en"].as<bool>();
        pcd_save_en_ = yaml["pcd_save"]["pcd_save_en"].as<bool>();
        pcd_save_interval_ = yaml["pcd_save"]["interval"].as<int>();
        extrinT_ = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["ivox_nearby_type"].as<int>();
    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    run_in_offline_ = true;
    return true;
}

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    // ROS subscribe initialization
    std::string lidar_topic, imu_topic;
    nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
    nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");

    if (preprocess_->GetLidarType() == LidarType::AVIA) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            lidar_topic, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });

    // USE_VIO
    std::string image_topic;
    nh.param<std::string>("common/img_topic", image_topic, "/sync_camera/color/image_raw");
    sub_img_ = nh.subscribe<sensor_msgs::Image>(image_topic, 200000,
                                              [this](const sensor_msgs::Image::ConstPtr &msg) { IMAGECallBack(msg); });

    // ROS publisher init
    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "camera_init";

    path_grav_.header.stamp = ros::Time::now();
    path_grav_.header.frame_id = "camera_init";

    path_fused_.header.stamp = ros::Time::now();
    path_fused_.header.frame_id = "camera_init";

    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_odom_aft_mapped_grav_ = nh.advertise<nav_msgs::Odometry>("/OdometryGrav", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
    pub_path_grav_ = nh.advertise<nav_msgs::Path>("/path_g", 100000);

    pub_odom_fused_ = nh.advertise<nav_msgs::Odometry>("/OdometryFused", 100000);
    pub_path_fused_ = nh.advertise<nav_msgs::Path>("/PathFused", 100000);
}

LaserMapping::LaserMapping() {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
}

void LaserMapping::Run() {
    if (!SyncPackages()) {
        return;
    }

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);
    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return;
    }

    // LOG(WARNING) << flg_LIO_optimized_;
    state_point_ = kf_.get_x();
    // 第一次 scan_undistort_gravity_ 是空的，就将scan_undistort_转到 gravity 系下 构建 scan_undistort_gravity_
    if (!flg_LIO_optimized_)   
    {
        double g_x = state_point_.grav[0];  // 这个是第一帧IMU系下的重力加速度
        double g_y = state_point_.grav[1];
        double g_z = state_point_.grav[2];
        common::V3D g_cur(-g_x,-g_y,-g_z);
        common::M3D R_g_i = Utility::g2R(g_cur);
        double yaw = Utility::R2ypr(R_g_i).x(); // 保持yaw角不变,绕pitch角和roll角旋转一个角度(主要是pitch角),所以这个yaw接近0
        R_g_i = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R_g_i; // 原来是-yaw
        R_grav_imu_ = R_g_i;
        // printf("!flg_LIO_optimized_ gravity %f %f %f\n", g_x, g_y, g_z);
        // common::V3D p_cur = state_point_.pos;
        // printf("!flg_LIO_optimized_ position %f %f %f\n", p_cur[0], p_cur[1], p_cur[2]);

        int scan_undistort_size = scan_undistort_->points.size();
        scan_undistort_gravity_.reset(new PointCloudType(scan_undistort_size, 1));
        for (int i = 0; i < scan_undistort_size; i++)
            PointBodyToWorldGravity(&scan_undistort_->points[i], &scan_undistort_gravity_->points[i]);
    }

    // 第一次跑到这里的时候 只有一帧点云 构造ivox之后就return了 
    // 第二次跑到这里的时候 还没有运行IEKF flg_LIO_optimized_ 还是 0 scan_undistort_gravity_ 用预测的状态来算世界系坐标
    // scan_undistort_gravity_ 保证是world(gravity)下的就好
    // TODO: 这里修改addImageCloudTrackFeature函数，返回特征跟踪的结果 std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> track_res;
    track_res = visual_sparse_map_->addImageCloudTrackFeature(measures_.meas_group_cam_.image_time_, measures_.meas_group_cam_.img_, scan_undistort_gravity_, 
                                                  RotCur(), PosCur(), R_grav_imu_);

    if(!p_imu_->IMUInitFlag() && !initialize_vio_gravity_)
    {
        auto g = kf_.get_x().grav;
        common::V3D lio_initial_g(g[0],g[1],g[2]);
        vins_estimator_->getGravity(lio_initial_g);
        initialize_vio_gravity_ = true;
    }

    // 为空闲的线程输入跟踪结果和IMU数据
    // 遍历 measures_.meas_group_cam_.cam_with_imu_ 放IMU
    auto imus = measures_.meas_group_cam_.cam_with_imu_;
    for (auto it_imu = imus.begin(); it_imu < imus.end(); it_imu++)
    {
        auto &&imu_meas = *(it_imu);
        double omega_x = imu_meas->angular_velocity.x;
        double omega_y = imu_meas->angular_velocity.y;
        double omega_z = imu_meas->angular_velocity.z;
        double a_x = imu_meas->linear_acceleration.x;
        double a_y = imu_meas->linear_acceleration.y;
        double a_z = imu_meas->linear_acceleration.z;
        double t = imu_meas->header.stamp.toSec();
        // std::cout<<"imu_t_: "<<t<<std::endl;
        Eigen::Vector3d acc(a_x,a_y,a_z);
        Eigen::Vector3d ang_v(omega_x,omega_y,omega_z);
        // vins_estimator_->inputIMU(t, acc, ang_v);
    }

    // 跟踪后把视觉的结果放进去
    
    tracker_letnet_->trackImageLetnet(measures_.meas_group_cam_.image_time_, measures_.meas_group_cam_.img_);
    vins_estimator_->inputFeature(measures_.meas_group_cam_.image_time_,track_res); // VINS没有开多线程的话，这里就跑一次 processMeasurements
    // if (SHOW_TRACK)
    // {
    //     cv::Mat imgTrack = tracker_letnet_->getTrackImage();
    //     pubTrackImage(imgTrack, measures_.meas_group_cam_.image_time_);
    // }

    /// the first scan
    if (flg_first_scan_) {
        ivox_->AddPoints(scan_undistort_->points);
        first_lidar_time_ = measures_.lidar_bag_time_;
        flg_first_scan_ = false;
        return;
    }
    flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= options::INIT_TIME;

    /// downsample
    Timer::Evaluate(
        [&, this]() {
            voxel_scan_.setInputCloud(scan_undistort_);
            voxel_scan_.filter(*scan_down_body_);
        },
        "Downsample PointCloud");

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    residuals_.resize(cur_pts, 0);
    point_selected_surf_.resize(cur_pts, true);
    plane_coef_.resize(cur_pts, common::V4F::Zero());

    // 这里是LIO的求解，VIO在另一个线程里头
    // ICP and iterated Kalman filter update
    Timer::Evaluate(
        [&, this]() {
            // iterated state estimation
            double solve_H_time = 0;
            // update the observation model, will call nn and point-to-plane residual computation
            kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV, solve_H_time);
            // save the state
            state_point_ = kf_.get_x();
            euler_cur_ = SO3ToEuler(state_point_.rot);
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
            flg_LIO_optimized_ = true;
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

    // scan_undistort_ 转到gravity系下 变成 scan_undistort_gravity_ 用于下一次运行构建视觉submap
    int scan_undistort_size = scan_undistort_->points.size();
    scan_undistort_gravity_.reset(new PointCloudType(scan_undistort_size, 1));
    for (int i = 0; i < scan_undistort_size; i++) {
        // PointBodyLidarToIMU(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
        PointBodyToWorldGravity(&scan_undistort_->points[i], &scan_undistort_gravity_->points[i]);
    }

    // std::cout<<"gravity: \n "<<state_point_.grav<<std::endl;
    // LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
            //   << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    // TODO: 我这里要取出VIO的求解结果然后和LIO叠加
    // nav_msgs::Odometry vio_odo = vins_estimator_->getOdometry();

    // publish or save map pcd
    if (run_in_offline_) {
        if (pcd_save_en_) {
            PublishFrameWorld();
        }
        if (path_save_en_) {
            PublishPath(pub_path_);
        }
    } else {
        if (pub_odom_aft_mapped_) {
            PublishOdometry(pub_odom_aft_mapped_);
        }
        if(pub_odom_aft_mapped_grav_){
            PublishOdometryGrav(pub_odom_aft_mapped_grav_); // 转换到和VIO一样？的重力系下
        }
        if (path_pub_en_ || path_save_en_) {
            PublishPath(pub_path_);
            PublishPathGrav(pub_path_grav_);    // 和VIO一样？的重力系下
        }
        if (scan_pub_en_ || pcd_save_en_) {
            PublishFrameWorld();
        }
        if (scan_pub_en_ && scan_body_pub_en_) {
            PublishFrameBody(pub_laser_cloud_body_);
        }
        if (scan_pub_en_ && scan_effect_pub_en_) {
            PublishFrameEffectWorld(pub_laser_cloud_effect_world_);
        }
        if(pub_path_fused_ && pub_odom_fused_)
        {
            PublishOdomPathFused(pub_odom_fused_,pub_path_fused_);
        }
    }

    send_odom_serial_.sendBySerial();   // 分别在 PublishOdometryGrav 和 SetPosestampGrav 函数中设置了要发送的时间和里程计信息

    // Debug variables
    frame_num_++;
}

void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.push_back(ptr);
            time_buffer_.push_back(msg->header.stamp.toSec());
            last_timestamp_lidar_ = msg->header.stamp.toSec();
        },
        "Preprocess (Standard)");
    mtx_buffer_.unlock();
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                LOG(WARNING) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            last_timestamp_lidar_ = msg->header.stamp.toSec();

            if (!time_sync_en_ && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
                !lidar_buffer_.empty()) {
                LOG(INFO) << "IMU and LiDAR not Synced, IMU time: " << last_timestamp_imu_
                          << ", lidar header time: " << last_timestamp_lidar_;
            }

            if (time_sync_en_ && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
                !imu_buffer_.empty()) {
                timediff_set_flg_ = true;
                timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
                LOG(INFO) << "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu_;
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.emplace_back(ptr);
            time_buffer_.emplace_back(last_timestamp_lidar_);
        },
        "Preprocess (Livox)");

    mtx_buffer_.unlock();
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
    publish_count_++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu_) > 0.1 && time_sync_en_) {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu_ + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer_.lock();
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.emplace_back(msg);
    mtx_buffer_.unlock();
}

// USE_VIO
cv::Mat LaserMapping::getImageFromMsg(const sensor_msgs::ImageConstPtr& img_msg) 
{
    cv::Mat img;
    img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
    // cv::imshow("img", img);
    // cv::waitKey(1);
    return img;
}

void LaserMapping::IMAGECallBack(const sensor_msgs::ImageConstPtr& msg)
{
    if (!img_en_) 
    {
        return;
    }
    // printf("[ INFO ]: get img at time: %.6f.\n", msg->header.stamp.toSec());
    if (msg->header.stamp.toSec() < last_timestamp_img_)
    {
        ROS_ERROR("img loop back, clear buffer");
        img_buffer_.clear();
        img_time_buffer_.clear();
    }
    mtx_buffer_.lock();
    img_buffer_.push_back(getImageFromMsg(msg));
    img_time_buffer_.push_back(msg->header.stamp.toSec());
    last_timestamp_img_ = msg->header.stamp.toSec();
    mtx_buffer_.unlock();
}

// bool LaserMapping::SyncPackages() {
//     if (lidar_buffer_.empty() || imu_buffer_.empty()) {
//         return false;
//     }

//     /*** push a lidar scan ***/
//     if (!lidar_pushed_) {
//         measures_.lidar_ = lidar_buffer_.front();
//         measures_.lidar_bag_time_ = time_buffer_.front();

//         if (measures_.lidar_->points.size() <= 1) {
//             LOG(WARNING) << "Too few input point cloud!";
//             lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
//         } else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_) {
//             lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
//         } else {
//             scan_num_++;
//             lidar_end_time_ = measures_.lidar_bag_time_ + measures_.lidar_->points.back().curvature / double(1000);
//             lidar_mean_scantime_ +=
//                 (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
//         }

//         measures_.lidar_end_time_ = lidar_end_time_;
//         lidar_pushed_ = true;
//     }

//     if (last_timestamp_imu_ < lidar_end_time_) {
//         return false;
//     }

//     /*** push imu_ data, and pop from imu_ buffer ***/
//     double imu_time = imu_buffer_.front()->header.stamp.toSec();
//     measures_.imu_.clear();
//     while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
//         imu_time = imu_buffer_.front()->header.stamp.toSec();
//         if (imu_time > lidar_end_time_) break;
//         measures_.imu_.push_back(imu_buffer_.front());
//         imu_buffer_.pop_front();
//     }

//     lidar_buffer_.pop_front();
//     time_buffer_.pop_front();
//     lidar_pushed_ = false;
//     return true;
// }

// TODO: 针对视觉要改这个同步的 做了硬件同步的话，这里取出了一张图像，一帧点云和相关联的IMU
bool LaserMapping::SyncPackages() {
    if (lidar_buffer_.empty() || imu_buffer_.empty() || img_buffer_.empty()) {
        // ROS_WARN("buffer empty!");
        return false;
    }

    /*** push a lidar scan ***/
    // 这里的 lidar_end_time_ 很重要，因为fast-lio的主要贡献就是 back-propogation
    // 最后的结果就是将这一帧的点云都校正到这一帧扫描结束的位置，对应 lidar_end_time_
    if (!lidar_pushed_) {
        // 第一个雷达测量（指向点云的首地址）
        measures_.lidar_ = lidar_buffer_.front();
        // 第一个雷达测量的时间戳
        measures_.lidar_bag_time_ = time_buffer_.front();

        // 每一帧点云数量大于1才有效
        if (measures_.lidar_->points.size() <= 1) {
            LOG(WARNING) << "Too few input point cloud!";
            // 记录无效帧的时间
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } 
        // 最后一个点的时间偏移不足 0.5*lidar_mean_scantime_ 说明雷达没有正常启动
        else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } 
        // 下面是正常启动的雷达
        else {
            scan_num_++;
            // 该雷达帧结束的时间戳 = 开始扫描的时间戳+扫描用时
            lidar_end_time_ = measures_.lidar_bag_time_ + measures_.lidar_->points.back().curvature / double(1000);
            // 雷达扫描一次的实际时间偏移
            lidar_mean_scantime_ +=
                (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
        }

        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

//  USE_VIO
    measures_.meas_group_cam_.img_ = img_buffer_.front().clone();
    measures_.meas_group_cam_.image_time_ = img_time_buffer_.front();

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    // 队列最前面IMU的时间戳
    double imu_time = imu_buffer_.front()->header.stamp.toSec();
    measures_.imu_.clear();
    measures_.meas_group_cam_.cam_with_imu_.clear();

    // 取一帧lidar前面的IMU数据
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = imu_buffer_.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time_) break;  // IMU的时间不能大于雷达扫描结束的时间 因为视觉是硬件同步的，图像时间和lidar时间差忽略不计
        measures_.imu_.push_back(imu_buffer_.front());
//  USE_VIO
        measures_.meas_group_cam_.cam_with_imu_.push_back(imu_buffer_.front());  // 与视觉相关联的IMU

        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
//  USE_VIO
    img_buffer_.pop_front();
    img_time_buffer_.pop_front();

    lidar_pushed_ = false;
    return true;
}

void LaserMapping::PrintState(const state_ikfom &s) {
    LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose()
              << ", off r: " << s.offset_R_L_I.coeffs().transpose() << ", t: " << s.offset_T_L_I.transpose();
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = common::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
            auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

            /** closest surface search and residual computation **/
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];

                /* transform to world frame */
                common::V3F p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                if (ekfom_data.converge) {
                    /** Find the closest surfaces in the map 点到平面 **/
                    ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                    point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                    if (point_selected_surf_[i]) {
                        point_selected_surf_[i] =
                            common::esti_plane(plane_coef_[i], points_near, options::ESTI_PLANE_THRESHOLD);
                    }
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    effect_feat_num_ = 0;

    // scan_down_body_ 是 scan_undistort_ 降采样后的，corr_pts_ 来自 scan_down_body_
    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            ekfom_data.h.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const common::M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
            const common::V3F off_t = s.offset_T_L_I.cast<float>();
            const common::M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                common::V3F point_this_be = corr_pts_[i].head<3>();
                common::M3F point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                common::V3F point_this = off_R * point_this_be + off_t;
                common::M3F point_crossmat = SKEW_SYM_MATRIX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                common::V3F norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                common::V3F C(Rt * norm_vec);
                common::V3F A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    common::V3F B(point_be_crossmat * off_R.transpose() * C);
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                        B[1], B[2], C[0], C[1], C[2];
                } else {
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0;
                }

                /*** Measurement: distance to the closest surface/corner ***/
                ekfom_data.h(i) = -corr_pts_[i][3];
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
}

/////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

////////////////// Publish Path /////////////////////////
void LaserMapping::PublishPath(const ros::Publisher pub_path) {
    SetPosestamp(msg_body_pose_);
    msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose_.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    path_.poses.push_back(msg_body_pose_);
    if (run_in_offline_ == false) {
        pub_path.publish(path_);
    }
}

void LaserMapping::PublishPathGrav(const ros::Publisher pub_path_grav) {
    SetPosestampGrav(msg_body_pose_grav_);
    msg_body_pose_grav_.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose_grav_.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    path_grav_.poses.push_back(msg_body_pose_grav_);
    if (run_in_offline_ == false) {
        pub_path_grav.publish(path_grav_);
    }
}

// void LaserMapping::PublishPathFuse(const ros::Publisher pub_path_fuse)
// {
    
// }


////////////////// Publish Odometry /////////////////////////
void LaserMapping::PublishOdometry(const ros::Publisher &pub_odom_aft_mapped) {
    odom_aft_mapped_.header.frame_id = "camera_init";
    odom_aft_mapped_.child_frame_id = "body";
    odom_aft_mapped_.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
    SetPosestamp(odom_aft_mapped_.pose);
    pub_odom_aft_mapped.publish(odom_aft_mapped_);
    auto P = kf_.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odom_aft_mapped_.pose.covariance[i * 6 + 0] = P(k, 3);
        odom_aft_mapped_.pose.covariance[i * 6 + 1] = P(k, 4);
        odom_aft_mapped_.pose.covariance[i * 6 + 2] = P(k, 5);
        odom_aft_mapped_.pose.covariance[i * 6 + 3] = P(k, 0);
        odom_aft_mapped_.pose.covariance[i * 6 + 4] = P(k, 1);
        odom_aft_mapped_.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odom_aft_mapped_.pose.pose.position.x, odom_aft_mapped_.pose.pose.position.y,
                                    odom_aft_mapped_.pose.pose.position.z));
    q.setW(odom_aft_mapped_.pose.pose.orientation.w);
    q.setX(odom_aft_mapped_.pose.pose.orientation.x);
    q.setY(odom_aft_mapped_.pose.pose.orientation.y);
    q.setZ(odom_aft_mapped_.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped_.header.stamp, tf_world_frame_, tf_imu_frame_));
}

void LaserMapping::PublishOdometryGrav(const ros::Publisher &pub_odom_grav_sys)
{
    odom_aft_mapped_grav_.header.frame_id = "camera_init";
    odom_aft_mapped_grav_.child_frame_id = "body";
    odom_aft_mapped_grav_.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
    send_odom_serial_.Time_ = lidar_end_time_;
    SetPosestampGrav(odom_aft_mapped_grav_.pose);
    lio_odom_queue_.push_back(odom_aft_mapped_grav_);
    pub_odom_grav_sys.publish(odom_aft_mapped_grav_);
}

// void LaserMapping::PublishOdometryFuse(const ros::Publisher &pub_odom_fuse)
// {
//     odom_fused_.header.frame_id = "camera_init";
//     odom_fused_.child_frame_id = "body";
//     odom_fused_.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
// }

////////////////// Publish Fuse /////////////////////////
void LaserMapping::PublishOdomPathFused(const ros::Publisher &pub_odom_fused, const ros::Publisher &pub_path_fused)
{
    
    odom_fused_.header.frame_id = "camera_init";
    odom_fused_.child_frame_id = "body";
    odom_fused_.header.stamp = ros::Time().fromSec(lidar_end_time_);
    SetPosestampFuse(odom_fused_.pose);
    pub_odom_fused.publish(odom_fused_);

    msg_body_pose_fused_.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose_fused_.header.frame_id = "camera_init";
    msg_body_pose_fused_.pose.position.x = odom_fused_.pose.pose.position.x;
    msg_body_pose_fused_.pose.position.y = odom_fused_.pose.pose.position.y;
    msg_body_pose_fused_.pose.position.z = odom_fused_.pose.pose.position.z;
    msg_body_pose_fused_.pose.orientation.x = odom_fused_.pose.pose.orientation.x;
    msg_body_pose_fused_.pose.orientation.y = odom_fused_.pose.pose.orientation.y;
    msg_body_pose_fused_.pose.orientation.z = odom_fused_.pose.pose.orientation.z;
    msg_body_pose_fused_.pose.orientation.w = odom_fused_.pose.pose.orientation.w;
    path_fused_.poses.push_back(msg_body_pose_fused_);

    if (run_in_offline_ == false)
        pub_path_fused.publish(path_fused_);

}

////////////////// Publish Frame /////////////////////////
void LaserMapping::PublishFrameWorld() {
    if (!(run_in_offline_ == false && scan_pub_en_) && !pcd_save_en_) {
        return;
    }

    PointCloudType::Ptr laserCloudWorld;
    // dense_publish_en: true   这里就是发布更稠密的点云，scan_undistort_ 就是没有降采样的那种，他比较稠密
    if (dense_pub_en_) {
        PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld.reset(new PointCloudType(size, 1));
        for (int i = 0; i < size; i++) {
            // PointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
            PointBodyToWorldGravity(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }
    } else {
        laserCloudWorld = scan_down_world_;
    }

    if (run_in_offline_ == false && scan_pub_en_) {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_world_.publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en_) {
        *pcl_wait_save_ += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
            pcd_index_++;
            std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/scans_") + std::to_string(pcd_index_) +
                                       std::string(".pcd"));
            pcl::PCDWriter pcd_writer;
            LOG(INFO) << "current scan saved to /PCD/" << all_points_dir;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
            pcl_wait_save_->clear();
            scan_wait_num = 0;
        }
    }
}

// 发布 scan_undistort_, 也就是每一帧lidar点经过IMU去畸变后
void LaserMapping::PublishFrameBody(const ros::Publisher &pub_laser_cloud_body) {
    // int size = scan_undistort_->points.size();
    // PointCloudType::Ptr laser_cloud_imu_body(new PointCloudType(size, 1));

    // for (int i = 0; i < size; i++) {
    //     // PointBodyLidarToIMU(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
    //     PointBodyToWorldGravity(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
    // }

    sensor_msgs::PointCloud2 laserCloudmsg;
    // pcl::toROSMsg(*laser_cloud_imu_body, laserCloudmsg);
    pcl::toROSMsg(*scan_undistort_gravity_, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "camera_init";
    // laserCloudmsg.header.frame_id = "body";
    pub_laser_cloud_body.publish(laserCloudmsg);
    publish_count_ -= options::PUBFRAME_PERIOD;
}

// 发布 corr_pts_, 也就是每一帧lidar点经过IMU去畸变后的结果，其实他还是body系下的帧
void LaserMapping::PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world) {
    int size = corr_pts_.size();
    PointCloudType::Ptr laser_cloud(new PointCloudType(size, 1));

    for (int i = 0; i < size; i++) {
        // PointBodyToWorld(corr_pts_[i].head<3>(), &laser_cloud->points[i]);
        PointBodyToWorldGravity(corr_pts_[i].head<3>(), &laser_cloud->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "camera_init";
    pub_laser_cloud_effect_world.publish(laserCloudmsg);
    publish_count_ -= options::PUBFRAME_PERIOD;
}

void LaserMapping::Savetrajectory(const std::string &traj_file) {
    std::ofstream ofs;
    ofs.open(traj_file, std::ios::out);
    if (!ofs.is_open()) {
        LOG(ERROR) << "Failed to open traj_file: " << traj_file;
        return;
    }

    ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
    for (const auto &p : path_.poses) {
        ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
            << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
            << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
    }

    ofs.close();
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////
template <typename T>
void LaserMapping::SetPosestamp(T &out) {
    out.pose.position.x = state_point_.pos(0);
    out.pose.position.y = state_point_.pos(1);
    out.pose.position.z = state_point_.pos(2);
    out.pose.orientation.x = state_point_.rot.coeffs()[0];
    out.pose.orientation.y = state_point_.rot.coeffs()[1];
    out.pose.orientation.z = state_point_.rot.coeffs()[2];
    out.pose.orientation.w = state_point_.rot.coeffs()[3];
}

// 将输出的位姿（以IMU的初始姿态为基准）转换到重力坐标系（前左天）
template <typename T>
void LaserMapping::SetPosestampGrav(T &out)
{
    double p_x = state_point_.pos(0);
    double p_y = state_point_.pos(1);
    double p_z = state_point_.pos(2);

    double q_x = state_point_.rot.coeffs()[0];
    double q_y = state_point_.rot.coeffs()[1];
    double q_z = state_point_.rot.coeffs()[2];
    double q_w = state_point_.rot.coeffs()[3];

    double g_x = state_point_.grav[0];  // 这个是第一帧IMU系下的重力加速度
    double g_y = state_point_.grav[1];
    double g_z = state_point_.grav[2];

    double v_x = state_point_.vel[0];
    double v_y = state_point_.vel[1];
    double v_z = state_point_.vel[2];

    common::V3D g_cur(-g_x,-g_y,-g_z);
    common::M3D R_g_i = Utility::g2R(g_cur);
    double yaw = Utility::R2ypr(R_g_i).x(); // 保持yaw角不变,绕pitch角和roll角旋转一个角度(主要是pitch角),所以这个yaw接近0
    // std::cout<<"👇g_cur gravity: \n "<<g_cur.transpose()<<std::endl;
    // std::cout<<"yaw: "<<yaw<<std::endl;
    R_g_i = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R_g_i; // 原来是-yaw

    R_grav_imu_ = R_g_i;

    Eigen::Quaterniond Q_i(q_w,q_x,q_y,q_z);   // Eigen中四元数赋值的顺序，实数w在首
    Q_i.normalized();
    // common::M3D R_i = Q_i.toRotationMatrix();
    common::V3D P_i = state_point_.pos;
    common::V3D V_i = state_point_.vel;

    
    common::V3D P_g = R_g_i*P_i;
    common::V3D V_g = R_g_i*V_i;
    Eigen::Quaterniond Q_g_i(R_g_i);
    Eigen::Quaterniond Q_g = Q_g_i*Q_i;
    // common::M3D R_g = R_g_i*R_i;

    out.pose.position.x = P_g[0];
    out.pose.position.y = P_g[1];
    out.pose.position.z = P_g[2];
    out.pose.orientation.x = Q_g.x();
    out.pose.orientation.y = Q_g.y();
    out.pose.orientation.z = Q_g.z();
    out.pose.orientation.w = Q_g.w();

    send_odom_serial_.Pos_ = P_g;
    send_odom_serial_.Vel_ = V_g;
    send_odom_serial_.Q_ = Q_g;
}

template <typename T>
void LaserMapping::SetPosestampFuse(T &out)
{
    nav_msgs::Odometry odom_lio;
    nav_msgs::Odometry odom_vio;
    if(!lio_odom_queue_.empty() && !vins_estimator_->vio_odom_queue_.empty())   // 从队列中取出来，对齐时间戳再叠加
    {
        nav_msgs::Odometry odom_lio = lio_odom_queue_.back();
        nav_msgs::Odometry odom_vio = vins_estimator_->vio_odom_queue_.back();

        double lio_t = odom_lio.header.stamp.toSec();
        double vio_t = odom_vio.header.stamp.toSec();

        // double t_diff = lio_t-vio_t;
        // std::cout<<"t_diff: "<<t_diff<<std::endl;

        lio_odom_queue_.pop_back();
        vins_estimator_->vio_odom_queue_.pop_back();

        // odom_lio.pose.pose.position.x;
        // odom_lio.pose.pose.position.y;
        // odom_lio.pose.pose.position.z;
        // odom_lio.pose.pose.orientation.x;
        // odom_lio.pose.pose.orientation.y;
        // odom_lio.pose.pose.orientation.z;
        // odom_lio.pose.pose.orientation.w;

        // odom_vio.pose.pose.position.x;
        // odom_vio.pose.pose.position.y;
        // odom_vio.pose.pose.position.z;
        // odom_vio.pose.pose.orientation.x;
        // odom_vio.pose.pose.orientation.y;
        // odom_vio.pose.pose.orientation.z;
        // odom_vio.pose.pose.orientation.w;

        // out.pose = odom_lio.pose.pose;  // 搞什么打通直接叠加这两个玩意儿是没什么意义的，纯粹是为了应付交差而浪费时间
        out.pose.orientation = odom_lio.pose.pose.orientation;
        // out.pose.position.x = odom_vio.pose.pose.position.x*0.2+odom_lio.pose.pose.position.x*0.8;
        // out.pose.position.y = odom_vio.pose.pose.position.y*0.2+odom_lio.pose.pose.position.y*0.8;
        // out.pose.position.z = odom_vio.pose.pose.position.z*0.2+odom_lio.pose.pose.position.z*0.8;
        out.pose.position = odom_lio.pose.pose.position;
    }
    else
    {
        out.pose.position.x = state_point_.pos(0);
        out.pose.position.y = state_point_.pos(1);
        out.pose.position.z = state_point_.pos(2);
        out.pose.orientation.x = state_point_.rot.coeffs()[0];
        out.pose.orientation.y = state_point_.rot.coeffs()[1];
        out.pose.orientation.z = state_point_.rot.coeffs()[2];
        out.pose.orientation.w = state_point_.rot.coeffs()[3];
    }

    // out.pose.position.x;
    // out.pose.position.y;
    // out.pose.position.z;
    // out.pose.orientation.x;
    // out.pose.orientation.y;
    // out.pose.orientation.z;
    // out.pose.orientation.w;
}

void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po) {
    common::V3D p_body(pi->x, pi->y, pi->z);
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po) {
    common::V3D p_body(pi.x(), pi.y(), pi.z());
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::PointBodyToWorldGravity(const PointType *pi, PointType *const po) {
    common::V3D p_body(pi->x, pi->y, pi->z);
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    common::V3D p_gravity = R_grav_imu_*p_global;
    po->x = p_gravity(0);
    po->y = p_gravity(1);
    po->z = p_gravity(2);

    // po->x = p_global(0);
    // po->y = p_global(1);
    // po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyToWorldGravity(const common::V3F &pi, PointType *const po) {
    common::V3D p_body(pi.x(), pi.y(), pi.z()); // body imu系
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    common::V3D p_gravity = R_grav_imu_*p_global;
    po->x = p_gravity(0);
    po->y = p_gravity(1);
    po->z = p_gravity(2);
    
    // po->x = p_global(0);
    // po->y = p_global(1);
    // po->z = p_global(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    common::V3D p_body_lidar(pi->x, pi->y, pi->z);  // world lidar系
    common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);   // world lidar系->world imu系

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

common::V3D LaserMapping::PosCur()
{
    return state_point_.pos;
}

common::M3D LaserMapping::RotCur()
{
    double q_x = state_point_.rot.coeffs()[0];
    double q_y = state_point_.rot.coeffs()[1];
    double q_z = state_point_.rot.coeffs()[2];
    double q_w = state_point_.rot.coeffs()[3];

    Eigen::Quaterniond Q_i(q_w,q_x,q_y,q_z);   // Eigen中四元数赋值的顺序，实数w在首
    Q_i.normalized();
    common::M3D R = Q_i.toRotationMatrix();
    return R;
}

void LaserMapping::Finish() {
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save_->size() > 0 && pcd_save_en_) {
        std::string file_name = std::string("scans.pcd");
        std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        LOG(INFO) << "current scan saved to /PCD/" << file_name;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
    }

    LOG(INFO) << "finish done";
}

// 和数据发送相关的结构体
SendOdometryBySerialTCP::SendOdometryBySerialTCP()
{
    // clientSocket_ = socket(PF_INET,SOCK_STREAM,0);
    // if(clientSocket_<0)
    //     printf("clientSocket_ error");
    // memset(&serverAddr_, 0, sizeof(serverAddr_));

    // serverAddr_.sin_family=AF_INET;
    // serverAddr_.sin_addr.s_addr=inet_addr("127.0.0.1");
    // serverAddr_.sin_port=htons(7789);

    // if(connect(clientSocket_, (struct sockaddr*)&serverAddr_, sizeof(serverAddr_))==-1) 
    //     printf("connect() error!");

    // fs_ = std::ofstream(out_pose_file_, std::ofstream::out | std::ofstream::trunc);
    // if (!fs_.is_open()) 
    //     std::cerr << "Error opening pose file "<< out_pose_file_ << std::endl;
    // fs_.close();
}

void SendOdometryBySerialTCP::sendBySerial()
{
    serial::Serial sp;  //创建一个serial类
    serial::Timeout to = serial::Timeout::simpleTimeout(100);   //创建timeout
    sp.setPort("/dev/ttyTHS0");     //设置要打开的串口名称
    sp.setBaudrate(115200); //设置串口通信的波特率
    sp.setTimeout(to);  //串口设置timeout

    try
    {
        sp.open();  //打开串口
    }
    catch(serial::IOException& e)
    {
        ROS_ERROR_STREAM("Unable to open port.");
        return ;
    }

    // 需要发送的数据
    double t_ms_double = Time_*1000;
    int64_t t = t_ms_double;  // 时间 整数微秒? 毫秒？
    float f_px = Pos_[0];    // 位置 
    float f_py = Pos_[1];
    float f_pz = Pos_[2];
    float f_vx = Vel_[0];    // 速度
    float f_vy = Vel_[1];
    float f_vz = Vel_[2];
    float f_qx = Q_.x();     // 姿态
    float f_qy = Q_.y();
    float f_qz = Q_.z();
    float f_qw = Q_.w();

    // std::cout<<"Pos_: "<<Pos_.transpose()<<std::endl;
    ROS_INFO("Time Position_: %f: [%f, %f, %f]", t_ms_double, f_px, f_py, f_pz);
    // 上面按照先旋转ｘ轴(0)，然后y轴(1)，最后z轴得到的角度，并不是传统意义上，zyx旋转的欧拉角。yaw pitch roll
    // 得到的ea向量，分别对应的是rz, ry, rx旋转角度，注意和下文的顺序对应
    // 另外这里得到的角度，归一化的范围有些问题，代码中的说明是 
    // The returned angles are in the ranges [0:pi]x[-pi:pi]x[-pi:pi].
    // common::M3D rx = Q_.toRotationMatrix();
    // common::V3D ea = rx.eulerAngles(2,1,0);
    // fs_ = std::ofstream(out_pose_file_, std::ofstream::out | std::ofstream::app);
    // if(fs_.is_open())
    // {
    //     fs_<<f_qx<<" ";
    //     fs_<<f_qy<<" ";
    //     fs_<<f_qz<<" ";
    //     fs_<<f_qw<<" ";
    //     fs_<<ea[0]<<" ";
    //     fs_<<ea[1]<<" ";
    //     fs_<<ea[2]<<" \n";
    // }
    // else
    //     printf("unable to open ~/yhp_code/faster_livo_ws/out_pose_file.txt \n");
    // fs_.close();

    // int64_t t = 1;  // 时间 整数微秒? 毫秒？
    // float f_px = 2;    // 位置 
    // float f_py = 3;
    // float f_pz = 4;
    // float f_vx = 5;    // 速度
    // float f_vy = 6;
    // float f_vz = 7;
    // float f_qx = 8;     // 姿态
    // float f_qy = 9;
    // float f_qz = 10;
    // float f_qw = 11;

    // 初始化内存空间 
    memset(s_buffer,0,sizeof(s_buffer));

    short int bitwise_sum = 0; // 按位求和的校验位

    int i;
    s_buffer[0] = 0xfe; // 起始位 1byte
    s_buffer[1] = 0xa1; // ID 1byte

    uint64_t tmp_i64;
    tmp_i64 = t;
    // std::cout<<"tmp_i64_t "<<tmp_i64<<std::endl;
    for(i=9;i>1;i--)    // 时间 8bytes 2-9 高位在前，先给后面的低位
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i64;  // 或低8位
        tmp_i64 >>= 8;    // 右移8位，移除低位的一个字节数据
        bitwise_sum+=s_buffer[i];   // 按位累加，用于校验
    }
    
    int32_t sum_check=0;//求和，用于校验 
    int32_t tmp_i32;    // 临时变量 用来*1000转换为毫米
    /*************位置xyz***************/
    tmp_i32 = f_px*1000;  // std::cout<<"; f_px "<<f_px<<" px (mm): "<<tmp_i32;
    for(i=13;i>9;i--)    // 位置x 4bytes 10-13 
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }
    tmp_i32 = f_py*1000; // std::cout<<"; f_py "<<f_py<<" py (mm): "<<tmp_i32;
    for(i=17;i>13;i--)    // 位置y 4bytes 14-17 
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }
    tmp_i32 = f_pz*1000; // std::cout<<"; f_pz "<<f_pz<<" pz (mm): "<<tmp_i32<<std::endl;
    for(i=21;i>17;i--)    // 位置z 4bytes 18-21 
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }
    /*************速度xyz***************/
    tmp_i32 = f_vx*1000;
    for(i=25;i>21;i--)    // 速度x 4bytes 22-25
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }
    tmp_i32 = f_vy*1000;
    for(i=29;i>25;i--)    // 速度y 4bytes 26-29 
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }
    tmp_i32 = f_vz*1000;
    for(i=33;i>29;i--)    // 速度z 4bytes 30-33 
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }

    /*************姿态四元数xyzw***************/
    tmp_i32 = f_qx*1000;
    for(i=37;i>33;i--)    // 姿态x 4bytes 34-37
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }
    tmp_i32 = f_qy*1000;
    for(i=41;i>37;i--)    // 姿态y 4bytes 38-41 
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }
    tmp_i32 = f_qz*1000;
    for(i=45;i>41;i--)    // 姿态z 4bytes 42-45 
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }
    tmp_i32 = f_qw*1000;
    for(i=49;i>45;i--)    // 姿态w 4bytes 46-49 
    {
        s_buffer[i] = 0;
        s_buffer[i] |= tmp_i32;  
        tmp_i32 >>= 8;    
        bitwise_sum+=s_buffer[i];
    }

    short int int16_tmp = bitwise_sum;
    for(i=51;i>49;i--) // 校验位 按位求和的结果 2bytes 50-51
    {
        s_buffer[i] = 0;
        s_buffer[i] |= int16_tmp;  
        int16_tmp >>= 8;    
    }

    // 定义好数组之后我们就可以给单片机发送数据
    sp.write(s_buffer,52);//两个参数，第一个参数是要发送的数据地址，第二个数据是要发送数据的长度
    // 关闭串口
    sp.close();
    // for(int tt=0;tt<52;tt++)
        // printf("%02x ",s_buffer[tt]);
    // std::cout<<std::endl;
    // printf("send .. ");
    // TCP 发送
    // send(clientSocket_,(const void*)s_buffer,52,0);
    

    return ;
}

}  // namespace faster_lio