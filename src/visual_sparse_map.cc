#include "visual_sparse_map.h"

VisualSubMap::VisualSubMap(const std::string &cam_calib_file, const std::string &model_param_str, const std::string &model_file_str)
{
    m_camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam_calib_file);
    w_ = COL;
    h_ = ROW;
    scale_ = 0.5;
    voxel_size_ = 0.5;
    letnet_w_ = w_*scale_;
    letnet_h_ = h_*scale_;
    score_thd_ = 30;

    max_dist_ = 40;
    max_point_num_ = 400;

    model_param_ = model_param_str;
    model_file_ = model_file_str;

    id_cnt_ = 0;

    letnet_init();
}

VisualSubMap::~VisualSubMap()
{

}

/**
 * @attention 重要! 构造视觉子地图并跟踪的函数
 * @param _t 时间戳 
 * @param _img 图像
 * @param _cloud_in 上一时刻的点云，世界系下的，位置已经优化了的。如果是头两帧的点云，他是没有ICP的，就用预测的位姿计算世界系下的坐标
 * @param _r_cur _p_cur _r_g_w 是当前帧到来时通过IMU预测的位置、姿态、重力方向(世界系)
*/
std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> VisualSubMap::addImageCloudTrackFeature(const double _t, const cv::Mat &_img, const PointCloudType::Ptr _cloud_in,
                                             const faster_lio::common::M3D &_r_cur, const faster_lio::common::V3D &_p_cur, 
                                             const faster_lio::common::M3D &_r_g_w)
{
    cur_time = _t;

    cv::cvtColor(_img, img_gray_, cv::COLOR_BGR2GRAY);

    TicToc t_r;
    letnet_convert(_img);
    // printf("Convert by Letnet costs: %fms \n", t_r.toc());

    cv::Mat depth_img = cv::Mat::zeros(h_, w_ , CV_32FC1);
    float* it = (float*)depth_img.data;

    voxels_current_.clear();
    cv::Mat score_norm;
    cv::normalize(score_,score_norm,0,255,cv::NORM_MINMAX,CV_8UC1); //score 在函数 letnet_convert 中转换
    // cv::imshow("score_map_norm",score_norm);
    // Step1: 从投影到当前帧中的点云找体素 遍历点云中的每一个点并投影到图像下
    // _cloud_in是重力 gravity 系，->world->body->camera
    std::vector<cv::Point2f>  candidate_projections;
    std::vector<float> candidate_scores;
    std::vector<float> candidate_depths;
    std::vector<cv::Point2f>  exist_projections;
    std::vector<cv::Point2f>  new_projections;
    std::vector<float> depths_form_lidar;
    Eigen::Isometry3d T_cam2lidar = Eigen::Isometry3d::Identity();
    T_cam2lidar.rotate(R_cl_);
    T_cam2lidar.pretranslate(t_cl_);
    Eigen::Isometry3d T_body2lidar = Eigen::Isometry3d::Identity();
    T_body2lidar.rotate(R_il_);
    T_body2lidar.pretranslate(t_il_);
    Eigen::Isometry3d T_world2body = Eigen::Isometry3d::Identity();
    T_world2body.rotate(_r_cur);
    T_world2body.pretranslate(_p_cur);
    Eigen::Isometry3d T_gravity2world = Eigen::Isometry3d::Identity();
    T_gravity2world.rotate(_r_g_w);
    T_gravity2world.pretranslate(Eigen::Vector3d::Zero());
    Eigen::Isometry3d T_cam2gravity = T_cam2lidar*T_body2lidar.inverse()*T_world2body.inverse()*T_gravity2world.inverse();  // 相机->LiDAR->body->world->gravity
    Eigen::Matrix3d R_c2g = T_cam2gravity.rotation();
    Eigen::Vector3d t_c2g = T_cam2gravity.translation();
    for(int i=0; i<_cloud_in->size(); i++)
    {
        Eigen::Vector3d pt_gravity(_cloud_in->points[i].x, _cloud_in->points[i].y, _cloud_in->points[i].z); // gravity 系下的坐标

        Eigen::Vector3d pt_camera = R_c2g*pt_gravity + t_c2g;
        Eigen::Vector2d uv_camera;
        m_camera_->spaceToPlane(pt_camera,uv_camera);    // 

        // gravity系下的点计算体素坐标
        int voxel_xyz[3];
        for (int j = 0; j < 3; j++)
            voxel_xyz[j] = floor(pt_gravity[j] / voxel_size_);

        // 成功投影到图像平面上并且深度大于0
        if(pt_camera[2]>0)
            if(in_border(uv_camera))
            {
                it[w_*int(uv_camera[1])+int(uv_camera[0])] = pt_camera[2];
                
                depths_form_lidar.push_back(pt_camera[2]);

                // Step2.1 符合一定条件的点 uv_camera，加入视觉 submap, 跟踪的操作是在缩放后的图像实现的
                Eigen::Vector2d uv_camera_scaled = uv_camera*scale_;
                float letnet_score = get_px_val(score_norm,uv_camera_scaled[0],uv_camera_scaled[1]);
                if(letnet_score > score_thd_)    // 深度大于0，投影到图像范围内，评分大于阈值
                {
                    VOXEL_KEY voxel_pos_current_point(voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]);  // 当前点体素
                    // 判断视觉submap中是否已存在该体素
                    auto existed_voxel = v_sub_map_.find(voxel_pos_current_point);
                    if(existed_voxel==v_sub_map_.end()) // 不存在，这是新体素，首帧普遍存在这个现象
                    {
                        POINTS_IN_VOXEL *pv = new POINTS_IN_VOXEL(0);
                        A_POINT_IN_VOXEL *apv = new A_POINT_IN_VOXEL(pt_gravity,letnet_score,pt_camera[2]);
                        pv->add_a_point(apv);
                        v_sub_map_[voxel_pos_current_point] = pv;
                        new_projections.push_back(cv::Point2f(uv_camera[0],uv_camera[1]));
                    }
                    else    // 该体素已存在，v_sub_map_可能包含历史数据，则加入新的点，需满足一定条件
                    {
                        exist_projections.push_back(cv::Point2f(uv_camera[0],uv_camera[1]));
                        if(letnet_score > existed_voxel->second->max_score_)
                        {
                            A_POINT_IN_VOXEL *apv = new A_POINT_IN_VOXEL(pt_gravity,letnet_score,pt_camera[2]);
                            existed_voxel->second->add_a_point(apv);
                            
                        }
                    }
                    voxels_current_.insert(voxel_pos_current_point);  // 当前帧的体素insert
                }
            }
        // if(pt_camera[2]<0)
        //     std::cout<<"pt_camera[2]: "<<pt_camera[2]<<std::endl;
    }
    cnt_++;
    // show_points_111(_img,new_projections,"points_new");
    // show_points_111(_img,exist_projections,"points_existed");

    cv::namedWindow("depth_img",cv::WINDOW_NORMAL);
    cv::resizeWindow("depth_img",640,512);
    cv::imshow("depth_img",depth_img);
    cv::waitKey(1);

    // Step 2: 遍历上面找到的所有体素(当前帧)
    for (auto &voxel_iter : voxels_current_)
    {
        VOXEL_KEY position = voxel_iter;  //; 这个体素的哈希值
        auto found_hash = v_sub_map_.find(position);
        if(found_hash!=v_sub_map_.end())
        {
            POINTS_IN_VOXEL *found_points_voxel = found_hash->second;    // 懒得写用迭代器了，直接auto
            for(auto &p_in_voxel : found_points_voxel->points_)
            {
                Eigen::Vector3d p_pos_grav = p_in_voxel->point_pos_;
                Eigen::Vector3d pt_camera_res = R_c2g*p_pos_grav + t_c2g;
                Eigen::Vector2d uv_camera_res;
                
                // 这边 candidate_depths candidate_scores candidate_projections都对应的
                candidate_depths.push_back(p_pos_grav[2]);  // 当前相机系下的深度
                candidate_scores.push_back(p_in_voxel->point_score_);
                m_camera_->spaceToPlane(pt_camera_res,uv_camera_res);    // 
                candidate_projections.push_back(cv::Point2f(uv_camera_res[0],uv_camera_res[1]));
            }
        }
        else
            ROS_WARN(" Something wrong when find hash in current image frame... ");
    }

    std::vector<cv::Point2f> added_pts;
    std::vector<float> add_depths;
    if(tracked_pixels_.empty() && tracked_point_depths_.empty())    // 首帧图像，仅提取不跟踪
    {
        std::vector<cv::Point3f> fts = feature_extractor(candidate_projections,candidate_scores,candidate_depths);
        for(auto &f:fts)
        {
            tracked_pixels_.push_back(cv::Point2f(f.x,f.y));
            tracked_point_depths_.push_back(f.z);
            ids_.push_back(id_cnt_++);
        }
    }
    else    // 其余情况，跟踪、提取
    {
        std::vector<cv::Point2f> tracked_pixels_candidate,
                                 tracked_pixels_new_scaled,
                                 tracked_pixels_new;
        std::vector<uchar> status;
        std::vector<float> err;

        tracked_pixels_candidate.resize(tracked_pixels_.size());
        for (int i = 0; i < int(tracked_pixels_.size()); i++)   // 先除以2 
        {
            tracked_pixels_candidate[i].x = tracked_pixels_[i].x * scale_;
            tracked_pixels_candidate[i].y = tracked_pixels_[i].y * scale_;
        }
        cv::calcOpticalFlowPyrLK(
                last_desc_,
                desc_,
                tracked_pixels_candidate,
                tracked_pixels_new_scaled,
                status,
                err);   // 在 desc_ (特征图)上实现跟踪，注意这里的desc_是缩放了的，缩小了一倍
        tracked_pixels_new.resize(tracked_pixels_new_scaled.size());
        for (int i = 0; i < int(tracked_pixels_new_scaled.size()); i++)   // 乘回去
        {
            tracked_pixels_new[i].x = tracked_pixels_new_scaled[i].x / scale_;
            tracked_pixels_new[i].y = tracked_pixels_new_scaled[i].y / scale_;
        }

        reduce_vector(ids_, status);
        reduce_vector(tracked_point_depths_, status);
        reduce_vector(tracked_pixels_, status);
        reduce_vector(tracked_pixels_new, status);

        // RANSAC 去除误匹配
        std::vector<uchar> ransac_status;
        cv::findFundamentalMat(tracked_pixels_, tracked_pixels_new, cv::FM_RANSAC, 1, 0.99, ransac_status);
        reduce_vector(ids_, ransac_status);
        reduce_vector(tracked_point_depths_, ransac_status);
        reduce_vector(tracked_pixels_, ransac_status);
        reduce_vector(tracked_pixels_new, ransac_status);
        // printf("tracked_pixels_ size: %li \n",tracked_pixels_.size());s
        ln_pts_velocity = ptsVelocity(ids_, tracked_pixels_, tracked_pixels_new);

        // show_points_111(_img,tracked_pixels_new,"tracked_pixels_");
        show_tracks_111(_img,tracked_pixels_,tracked_pixels_new,"tracked_pixels_");
        // TODO: 构造VINS能用的数据结构 std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> 并设成返回值
        //                                      |- 特征ID  |               |-相机ID，左目0，右目1
        //                                                |-在VINS里，单目的话这个 vector 的size是1，双目的话就是2，在这里建议vector的size是2，1保留跟踪结果，2里面放深度

        prev_time = cur_time;

        for (int i = 0; i < ids_.size(); i++)
        {
            int feature_id = ids[i];
            double x, y ,z,p_u, p_v,velocity_x, velocity_y;
            int flag = 0; // 1 for tracking result
            x = tracked_pixels_[i].x;
            y = tracked_pixels_[i].y;
            z = 1;
            p_u = tracked_pixels_new[i].x;
            p_v = tracked_pixels_new[i].y;
            velocity_x = ln_pts_velocity[i].x;
            velocity_y = ln_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> ln_xyz_uv_velocity;
            ln_xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            lnmap_[feature_id].emplace_back(flag,  ln_xyz_uv_velocity);
        }
 /*    
        for (int i = 0; i < ids_.size(); i++)
        {
            int feature_id = ids[i];
            double x, y ,z,p_u, p_v,velocity_x, velocity_y;
            int flag = 0; // 1 for tracking result
            x = 0;
            y = 0;
            z = 0;
            p_u = 0;
            p_v = 0;
            velocity_x = 0;
            velocity_y = 0;

            Eigen::Matrix<double, 7, 1> ln_xyz_uv_velocity;
            ln_xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            lnmap_[feature_id].emplace_back(flag,  ln_xyz_uv_velocity);
        }


*/
        // 提取新的点
        std::vector<cv::Point3f> fts_new = feature_extractor(candidate_projections,candidate_scores,
                                                             candidate_depths,tracked_pixels_new);

        // 数据迭代 tracked_points_ 用于下次的跟踪，需要迭代到下一次的点，分成两部分
        vector<float> d_new;
        vector<cv::Point2f> pixels_new;
        for(auto &f:fts_new)
        {
            pixels_new.push_back(cv::Point2f(f.x,f.y));
            d_new.push_back(f.z);
            ids_.push_back(id_cnt_++);
        }

        
        tracked_pixels_.clear();
        tracked_pixels_.insert(tracked_pixels_.end(), tracked_pixels_new.begin(), tracked_pixels_new.end());  // 上一帧跟过来的点
        tracked_pixels_.insert(tracked_pixels_.end(), pixels_new.begin(), pixels_new.end());  // 新增的点

    }

    // show_points_111(_img,candidate_projections,"candidate_projections");

    last_desc_ = desc_;
    return ln_map_;
}

std::vector<cv::Point3f> VisualSubMap::feature_extractor(const std::vector<cv::Point2f>& projections,const std::vector<float>& scores_from_letnet,   
                                                         const std::vector<float>& depths_form_lidar,const std::vector<cv::Point2f>& exist_points)
{
    std::vector<cv::Point3f> selected_fts;   // 像素坐标uv 以及对应的深度
    std::vector<int> indices;

    //  从三个条件去判断：
    //       1. 评分最大
    //       2. 深度连续性
    //       3. 每个特征点的间距不能太小

    // 排序, scores 从评分从高到低
    indices = sort_and_return_indices(scores_from_letnet);
    
    // scores 从高到低 遍历 评分并取出对应点 这里相当于遍历投影到图像当中的所有点
    for(int j=0;j<indices.size();j++)
    {
        cv::Point2f pt_candidate = projections[indices[j]];
        float depth_candidate = depths_form_lidar[indices[j]];
        bool keep = true;
        if(exist_points.empty())    // 第一帧,还没有提取到点
        {
            for(int k=0;k<selected_fts.size();k++)   // 和已经提到的点相比距离不能太近
            {
                cv::Point2f pt_selected(selected_fts[k].x,selected_fts[k].y);
                double d = cv::norm(pt_candidate-pt_selected);
                
                if(d<max_dist_)
                {
                    keep = false;
                    break;
                }
            }
            if(keep)
            {
                // res_depths.push_back(depth);
                // res_uvs.push_back(pt);
                selected_fts.push_back(cv::Point3f(pt_candidate.x,pt_candidate.y,depth_candidate));
            }
            if(selected_fts.size()>max_point_num_)
                break;
        } 
        else    // 图像中已经有点了
        {
            for(int k=0;k<exist_points.size();k++)  // 和已有的点对比距离
            {
                double d = cv::norm(pt_candidate-exist_points[k]);
                
                if(d<max_dist_)
                {
                    keep = false;
                    break;
                }
            }
            for(int l=0;l<selected_fts.size();l++)   // 还要和新提到的新的点对比距离
            {
                cv::Point2f pt_selected(selected_fts[l].x,selected_fts[l].y);
                double d = cv::norm(pt_candidate-pt_selected);
                
                if(d<max_dist_)
                {
                    keep = false;
                    break;
                }
            }
            if(keep)
            {
                // res_depths.push_back(depth);
                // res_uvs.push_back(pt);
                selected_fts.push_back(cv::Point3f(pt_candidate.x,pt_candidate.y,depth_candidate));
            }
            if((selected_fts.size()+exist_points.size())>max_point_num_)
                break;
        }
    }
    // cv::cornerSubPix
    return selected_fts;
}

void VisualSubMap::letnet_init()
{
    score_ = cv::Mat(letnet_h_, letnet_w_, CV_32FC1);
    desc_ = cv::Mat(letnet_h_, letnet_w_, CV_8UC3);
    last_desc_ = cv::Mat(letnet_h_, letnet_w_, CV_8UC3);
    // std::string model_param_str = std::string(ROOT_DIR) + "model/model.param";
    // std::string model_file_str = std::string(ROOT_DIR) + "model/model.bin";
    const char* model_param = model_param_.c_str();
    const char* model_file = model_file_.c_str();
    std::cout<<"net param: "<<model_param_<<std::endl;
    std::cout<<"net model: "<<model_file_<<std::endl;
    net_.opt.use_vulkan_compute = true;
    net_.opt.num_threads = 1;
    net_.load_param(model_param);
    net_.load_model(model_file);
}

void VisualSubMap::letnet_convert(const cv::Mat& image_bgr)
{
    last_desc_ = desc_.clone();
    cv::Mat img;
    cv::resize(image_bgr, img, cv::Size(letnet_w_, letnet_h_));
    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    // opencv to ncnn
    in_ = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    in_.substract_mean_normalize(mean_vals_, norm_vals_);
    // extract
    ex.input("input", in_);
    ex.extract("score", out1_); 
    ex.extract("descriptor", out2_);
    // ncnn to opencv
    out1_.substract_mean_normalize(mean_vals_inv_, norm_vals_inv_);
    out2_.substract_mean_normalize(mean_vals_inv_, norm_vals_inv_);

//    out1.to_pixels(score.data, ncnn::Mat::PIXEL_GRAY);
    memcpy((uchar*)score_.data, out1_.data, letnet_h_*letnet_w_*sizeof(float));
    cv::Mat desc_tmp(letnet_h_, letnet_w_, CV_8UC3);
    out2_.to_pixels(desc_tmp.data, ncnn::Mat::PIXEL_BGR);
    desc_ = desc_tmp.clone();
    // cv::imwrite("desc.png", desc_);
    // cv::imshow("desc_map.png",desc_);
}

float VisualSubMap::get_px_val(const cv::Mat &img, float x, float y) {
    // boundary check 检测边界
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    float xx = x - floor(x);// floor(x)表示对x向下取整。
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    // 就是双线性插值，4个点，离哪个点越近权重越大，总权重为1
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
            + xx * (1 - yy) * img.at<uchar>(y, x_a1)
            + (1 - xx) * yy * img.at<uchar>(y_a1, x)
            + xx * yy * img.at<uchar>(y_a1, x_a1);
}

bool VisualSubMap::in_border(const Eigen::Vector2d &pt)
{
    const int BORDER_SIZE = 3;
    int img_x = cvRound(pt[0]);
    int img_y = cvRound(pt[1]);
    return BORDER_SIZE <= img_x && img_x < w_ - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < h_ - BORDER_SIZE;
}

void VisualSubMap::setLidarCameraExtrinsic(const Eigen::Vector3d &T, const Eigen::Matrix3d &R)
{
    R_cl_ = R;
    t_cl_ = T;
}

void VisualSubMap::setIMULidarExtrinsic(const Eigen::Vector3d &T, const Eigen::Matrix3d &R)
{
    R_il_ = R;
    t_il_ = T;
}

bool VisualSubMap::compare(const std::pair<float,int>& a, const std::pair<float,int>& b)
{
    return a.first > b.first;
}

std::vector<int> VisualSubMap::sort_and_return_indices(const std::vector<float>& input)
{
    std::vector<std::pair<float,int>> pairs;
    for(int i=0;i<input.size();i++)
        pairs.push_back(std::make_pair(input[i],i));
    std::sort(pairs.begin(),pairs.end(),compare);
    std::vector<int> indices;
    for(const auto& pair:pairs)
        indices.push_back(pair.second);
    return indices;
}

void VisualSubMap::show_points_111(const cv::Mat& image, const std::vector<cv::Point2f> &pts, const std::string win_name)
{
    cv::Mat im = image.clone();
    for(int i=0; i<pts.size(); i++)
    {
        cv::Point2f p = pts[i];
        cv::circle(im, p, 3, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }

    // if(cnt_>=0&&cnt_<10)
    // {
    //     cv::imwrite("/home/casia/yhp_code/faster_livo_ws/"+std::to_string(cnt_)+".png",im);
    //     cv::Mat score_uchar;
    //     score_.convertTo(score_uchar,CV_8UC1,255.0);
    //     cv::imwrite("/home/casia/yhp_code/faster_livo_ws/score_"+std::to_string(cnt_)+".png",score_uchar);
    // }
        
    cv::namedWindow(win_name,cv::WINDOW_NORMAL);
    cv::imshow(win_name,im);
}

void VisualSubMap::show_tracks_111(const cv::Mat& image, const std::vector<cv::Point2f> &pxs_prev, const std::vector<cv::Point2f> &pxs_curr, const std::string win_name)
{
    cv::Mat im = image.clone();
    for(int i=0; i<pxs_prev.size(); i++)
    {
        cv::Point2f p1 = pxs_prev[i];
        cv::Point2f p2 = pxs_curr[i];
        cv::arrowedLine(im, p1, p2, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        cv::circle(im, p2, 6, cv::Scalar(255, 255, 0), -1, 8);
    }
        
    cv::namedWindow(win_name,cv::WINDOW_NORMAL);
    cv::imshow(win_name,im);
}


std::vector<cv::Point2f> VisualSubMap::ptsvelocity(std::vector<int> &ids, std::vector<cv::Point2f> &pxs_prev, const std::vector<cv::Point2f> &pxs_curr)
{
    std::vector<cv::Point2f> pts_velocity;
    std::map<int, cv::Point2f> &cur_id_pts;
    std::map<int, cv::Point2f> &pre_id_pts;

    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pxs_curr[i]));
        pre_id_pts.insert(make_pair(ids[i], pxs_prev[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pxs_prev.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pxs_curr[i].x - it->second.x) / dt;
                double v_y = (pxs_curr[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < pxs_curr.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}