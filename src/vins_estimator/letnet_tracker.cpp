#include "vins_estimator/letnet_tracker.h"

// 原1280*1024
#define LET_WIDTH 640 // 640
#define LET_HEIGHT 512 // 512

int FeatureTrackerLetnet::n_id = 0;
struct greaterThanPtr
{
    bool operator () (const float * a, const float * b) const
    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

bool FeatureTrackerLetnet::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

FeatureTrackerLetnet::FeatureTrackerLetnet()
{
    #ifdef LET_NET
    let_init();
    #else

    #endif
}


void FeatureTrackerLetnet::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// void FeatureTrackerLetnet::addPoints()
// {
//     for (auto &p : n_pts)
//     {
//         forw_pts.push_back(p);
//         ids.push_back(-1);
//         track_cnt.push_back(1);
//     }
// }

#if 0
void FeatureTrackerLetnet::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

#ifndef LET_NET
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;
#else
    img = _img;
    let_net(img);
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
#endif

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
#ifdef LET_NET
        std::vector<cv::Point2f> corners1, corners2;
        int w0 = forw_img.cols; int h0 = forw_img.rows;
        float k_w = float(w0) / float(LET_WIDTH);
        float k_h = float(h0) / float(LET_HEIGHT);
        // resize cur_pts to corners1
        corners1.resize(cur_pts.size());
        for (int i = 0; i < int(cur_pts.size()); i++)
        {
            corners1[i].x = cur_pts[i].x / k_w;
            corners1[i].y = cur_pts[i].y / k_h;
        }
        cv::calcOpticalFlowPyrLK(last_desc, desc, corners1, corners2,
                                 status,err,cv::Size(21, 21),5);
        // resize corners2 to forw_pts
        forw_pts.resize(corners2.size());
        for (int i = 0; i < int(corners2.size()); i++)
        {
            forw_pts[i].x = corners2[i].x * k_w;
            forw_pts[i].y = corners2[i].y * k_h;
        }
        // subpixel refinement
        cv::cornerSubPix(gray,
                         forw_pts,
                         cv::Size(3, 3),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                         5, 0.01));
#else
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
#endif
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {

#ifdef LET_NET
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            cv::resize(mask, mask, cv::Size(LET_WIDTH, LET_HEIGHT));
            letFeaturesToTrack(score, n_pts, MAX_CNT - forw_pts.size(), 0.0001, MIN_DIST, mask);
            int w0 = forw_img.cols; int h0 = forw_img.rows;
            float k_w = float(w0) / float(LET_WIDTH);
            float k_h = float(h0) / float(LET_HEIGHT);
            for (auto & n_pt : n_pts)
            {
                n_pt.x *= k_w;
                n_pt.y *= k_h;
            }
            // subpixel refinement
            cv::cornerSubPix(gray,
                             n_pts,
                             cv::Size(3, 3),
                             cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                             5, 0.01));
#else
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
#endif
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}
#endif

// 我这边就懒得加宏定义判断了，不懂就去看前面的函数好了
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTrackerLetnet::trackImageLetnet(double _cur_time, const cv::Mat &_img)
{
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img.clone(); // 这里一般用RGB图像
    row = cur_img.rows;
    col = cur_img.cols;

    let_net(cur_img);   //新来的图像转换成特征图和score_map，这里面resize了以下

    cv::cvtColor(cur_img, gray, cv::COLOR_BGR2GRAY);

    cur_pts.clear();

    // 上一帧有点就先跟踪
    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        std::vector<cv::Point2f> corners1, corners2;
        int w0 = cur_img.cols; int h0 = cur_img.rows;
        float k_w = float(w0) / float(LET_WIDTH);   // 2
        float k_h = float(h0) / float(LET_HEIGHT);
        // resize cur_pts to corners1
        corners1.resize(prev_pts.size());
        for (int i = 0; i < int(prev_pts.size()); i++)   // 先除以2 
        {
            corners1[i].x = prev_pts[i].x / k_w;
            corners1[i].y = prev_pts[i].y / k_h;
        }
        cv::calcOpticalFlowPyrLK(last_desc, desc, corners1, corners2,
                                 status,err,cv::Size(21, 21),5);    // 跟踪后
        // resize corners2 to cur_pts
        cur_pts.resize(corners2.size());
        for (int i = 0; i < int(corners2.size()); i++)  //再乘以2
        {
            cur_pts[i].x = corners2[i].x * k_w;
            cur_pts[i].y = corners2[i].y * k_h;
        }
        // subpixel refinement 亚像素优化
        cv::cornerSubPix(gray,
                         cur_pts,
                         cv::Size(3, 3),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                         5, 0.01));

        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (1)
    {
        // rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)  //跟踪之后从letnet的scoremap中提取特征点
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != cur_img.size())
                cout << "wrong size " << endl;

            cv::resize(mask, mask, cv::Size(LET_WIDTH, LET_HEIGHT));
            letFeaturesToTrack(score, n_pts, MAX_CNT - cur_pts.size(), 0.0001, MIN_DIST, mask);
            int w0 = cur_img.cols; int h0 = cur_img.rows;
            float k_w = float(w0) / float(LET_WIDTH);
            float k_h = float(h0) / float(LET_HEIGHT);
            for (auto & n_pt : n_pts)
            {
                n_pt.x *= k_w;
                n_pt.y *= k_h;
            }
            // subpixel refinement
            cv::cornerSubPix(gray,
                             n_pts,
                             cv::Size(3, 3),
                             cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                             5, 0.01));
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
    }

    cur_un_pts = undistortedPts(cur_pts, m_camera);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // 这里绝壁不用双目模式，不管他了
    cv::Mat empty_mat;
    vector<cv::Point2f> cur_right_pts_empty;
    if(SHOW_TRACK)
        drawTrack(cur_img, empty_mat, ids, cur_pts, cur_right_pts_empty, prevLeftPtsMap);

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }    

    // printf("Track Feature Letnet costs: %fms \n", t_r.toc());

    return featureFrame;
}

void FeatureTrackerLetnet::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTrackerLetnet::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTrackerLetnet::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTrackerLetnet::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTrackerLetnet::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

vector<cv::Point2f> FeatureTrackerLetnet::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTrackerLetnet::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTrackerLetnet::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols; 
    // if (!imRight.empty() && stereo_cam)
    //     cv::hconcat(imLeft, imRight, imTrack);
    // else
    //     imTrack = imLeft.clone();
    imTrack = imLeft.clone();
    // cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    // if (!imRight.empty() && stereo_cam)
    // {
    //     for (size_t i = 0; i < curRightPts.size(); i++)
    //     {
    //         cv::Point2f rightPt = curRightPts[i];
    //         rightPt.x += cols;
    //         cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
    //         //cv::Point2f leftPt = curLeftPtsTrackRight[i];
    //         //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
    //     }
    // }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    // cv::imshow("Letnet imTrack",imTrack); 
    cv::waitKey(1);
    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

void FeatureTrackerLetnet::letFeaturesToTrack(cv::InputArray image,
                                             cv::OutputArray _corners,
                                             int maxCorners,
                                             double qualityLevel,
                                             double minDistance,
                                             cv::InputArray _mask, int blockSize)
{
    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(_mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(image)));

    cv::Mat eig = image.getMat(), tmp;
    double maxVal = 0;
    cv::minMaxLoc(eig, 0, &maxVal, 0, 0, _mask);    // 
    cv::threshold(eig, eig, maxVal * qualityLevel, 0, cv::THRESH_TOZERO);
    cv::dilate(eig, tmp, cv::Mat());

    cv::Size imgsize = eig.size();
    std::vector<const float*> tmpCorners;

    cv::Mat Mask = _mask.getMat();
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )
                tmpCorners.push_back(eig_data + x);
        }
    }

    std::vector<cv::Point2f> corners;
    std::vector<float> cornersQuality;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (total == 0)
    {
        _corners.release();
        return;
    }

    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());

    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = eig.cols;
        int h = eig.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <cv::Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                cornersQuality.push_back(*tmpCorners[i]);

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            cornersQuality.push_back(*tmpCorners[i]);

            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            corners.push_back(cv::Point2f((float)x, (float)y));
            ++ncorners;

            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }

    cv::Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}

void FeatureTrackerLetnet::let_init(){
    score = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_32FC1);
    desc = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    last_desc = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    // std::string(std::string(ROOT_DIR) + "config/" + vins_config_
    // net.load_param("/home/casia/yhp_code/faster_livo_ws/src/faster-lio/model/model.param");
    // net.load_model("/home/casia/yhp_code/faster_livo_ws/src/faster-lio/model/model.bin");
    std::string model_param_str = std::string(ROOT_DIR) + "model/model.param";
    std::string model_file_str = std::string(ROOT_DIR) + "model/model.bin";
    const char* model_param = model_param_str.c_str();
    const char* model_file = model_file_str.c_str();
    std::cout<<"net param: "<<std::string(ROOT_DIR) + "model/model.param"<<std::endl;
    std::cout<<"net model: "<<std::string(ROOT_DIR) + "model/model.bin"<<std::endl;
    net.opt.use_vulkan_compute = true;
    net.opt.num_threads = 1;
    net.load_param(model_param);
    net.load_model(model_file);
}

void FeatureTrackerLetnet::let_net(const cv::Mat& image_bgr) {
    last_desc = desc.clone();
    cv::Mat img;
    cv::resize(image_bgr, img, cv::Size(LET_WIDTH, LET_HEIGHT));
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    // opencv to ncnn
    in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    in.substract_mean_normalize(mean_vals, norm_vals);
    // extract
    ex.input("input", in);
    ex.extract("score", out1);
    ex.extract("descriptor", out2);
    // ncnn to opencv
    out1.substract_mean_normalize(mean_vals_inv, norm_vals_inv);
    out2.substract_mean_normalize(mean_vals_inv, norm_vals_inv);

//    out1.to_pixels(score.data, ncnn::Mat::PIXEL_GRAY);
    memcpy((uchar*)score.data, out1.data, LET_HEIGHT*LET_WIDTH*sizeof(float));
    cv::Mat desc_tmp(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    out2.to_pixels(desc_tmp.data, ncnn::Mat::PIXEL_BGR);
    desc = desc_tmp.clone();
    cv::imwrite("desc.png", desc);
}

cv::Mat FeatureTrackerLetnet::getTrackImage()
{
    return imTrack;
}