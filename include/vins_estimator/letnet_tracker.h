#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "vins_estimator/parameters.h"
#include "vins_estimator/tic_toc.h"

// add
#include "net.h"

#include "feature_tracker.h"

#define LET_NET

using namespace std;
using namespace camodocal;
using namespace Eigen;

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);
// bool inBorder(const cv::Point2f &pt);

class FeatureTrackerLetnet
{
  public:
    FeatureTrackerLetnet();

    void readImage(const cv::Mat &_img,double _cur_time);

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImageLetnet(double _cur_time, const cv::Mat &_img);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);

    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);

    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                vector<int> &curLeftIds,
                                vector<cv::Point2f> &curLeftPts, 
                                vector<cv::Point2f> &curRightPts,
                                map<int, cv::Point2f> &prevLeftPtsMap);
    cv::Mat getTrackImage();

    void letFeaturesToTrack(cv::InputArray image,
                            cv::OutputArray corners,
                            int maxCorners,
                            double qualityLevel,
                            double minDistance,
                            cv::InputArray mask, int blockSize=3);
    void let_init();
    void let_net(const cv::Mat& image_bgr);

    bool inBorder(const cv::Point2f &pt);
    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, gray;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    // map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
    bool hasPrediction;

    // add
    cv::Mat score;
    cv::Mat desc;
    cv::Mat last_desc;
    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    const float mean_vals_inv[3] = {0, 0, 0};
    const float norm_vals_inv[3] = {255.f, 255.f, 255.f};
    ncnn::Net net;
    ncnn::Mat in;
    ncnn::Mat out1, out2;
};