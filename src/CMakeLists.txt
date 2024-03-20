add_library(${PROJECT_NAME}
        laser_mapping.cc
        pointcloud_preprocess.cc
        options.cc
        utils.cc
        visual_sparse_map.cc

        vins_estimator/utility.cpp
        vins_estimator/marginalization_factor.cpp
        vins_estimator/pose_local_parameterization.cpp
        vins_estimator/projectionOneFrameTwoCamFactor.cpp
        vins_estimator/projectionTwoFrameOneCamFactor.cpp
        vins_estimator/projectionTwoFrameTwoCamFactor.cpp
        vins_estimator/parameters.cpp
        vins_estimator/feature_tracker.cpp
        vins_estimator/letnet_tracker.cpp
        vins_estimator/initial_aligment.cpp
        vins_estimator/initial_ex_rotation.cpp
        vins_estimator/initial_sfm.cpp
        vins_estimator/solve_5pts.cpp
        vins_estimator/feature_manager.cpp
        vins_estimator/CameraPoseVisualization.cpp
        vins_estimator/visualization.cpp
        vins_estimator/estimator.cpp

        camodocal/chessboard/Chessboard.cc
        camodocal/calib/CameraCalibration.cc
        camodocal/camera_models/Camera.cc
        camodocal/camera_models/CameraFactory.cc
        camodocal/camera_models/CostFunctionFactory.cc
        camodocal/camera_models/PinholeCamera.cc
        camodocal/camera_models/PinholeFullCamera.cc
        camodocal/camera_models/CataCamera.cc
        camodocal/camera_models/EquidistantCamera.cc
        camodocal/camera_models/ScaramuzzaCamera.cc
        camodocal/sparse_graph/Transform.cc
        camodocal/gpl/gpl.cc
        camodocal/gpl/EigenQuaternionParameterization.cc
        )

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencpp livox_ros_driver_gencpp)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${CERES_LIBRARIES}
        tbb
        glog
        yaml-cpp
        ncnn
        )

target_include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
