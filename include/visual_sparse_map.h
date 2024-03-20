#ifndef VISUAL_SUB_MAP_H
#define VISUAL_SUB_MAP_H

#include "common_lib.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "vins_estimator/parameters.h"
#include "vins_estimator/tic_toc.h"
// #include "vins_estimator/letnet_tracker.h"
#include "net.h"

#define HASH_P 116101
#define MAX_N 10000000000


// 我的代码风格：重要的数据类型，需要频繁操作的，用大写；用来处理数据结构的功能类，大小写组合；类成员函数形参在前面加下划线；成员变量后面加下划线。变量名尽量避免重复。

class A_POINT_IN_VOXEL
{
public:
    A_POINT_IN_VOXEL(const Eigen::Vector3d &_p, float _score, float _d): point_pos_(_p),point_score_(_score),ref_depth_(_d)  {}
    Eigen::Vector3d point_pos_;
    float point_score_;
    float ref_depth_;
};

class POINTS_IN_VOXEL
{
public:
    POINTS_IN_VOXEL(u_int32_t _n):point_num_(_n),max_score_(0)   {}
    u_int32_t point_num_;
    u_int32_t max_score_;
    std::vector<A_POINT_IN_VOXEL*> points_; 
    void add_a_point(A_POINT_IN_VOXEL *_p)
    {   
        if(_p->point_score_>max_score_)
        {
            max_score_ = _p->point_score_;
            points_.push_back(_p); 
            point_num_++;
        }
        // points_.push_back(_p); 
        // point_num_++;  
    }
};

// Key of hash table
class VOXEL_KEY
{
public:
    int64_t x;
    int64_t y;
    int64_t z;
    VOXEL_KEY(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}
    bool operator==(const VOXEL_KEY &other) const
    {
        return (x == other.x && y == other.y && z == other.z);
    }
    bool operator<(const VOXEL_KEY &p) const
    {
        if (x < p.x)
            return true;
        if (x > p.x)
            return false;
        if (y < p.y)
            return true;
        if (y > p.y)
            return false;
        if (z < p.z)
            return true;
        if (z > p.z)
            return false;
    }
};

/**
 * s特化了std::hash模板，以便能够正确地计算VOXEL_KEY类型对象的哈希值
*/
namespace std
{
    template <>
    struct hash<VOXEL_KEY>
    {
        size_t operator()(const VOXEL_KEY &s) const
        {
            using std::hash;
            using std::size_t;
            return (((hash<int64_t>()(s.z) * HASH_P) % MAX_N + hash<int64_t>()(s.y)) * HASH_P) % MAX_N + hash<int64_t>()(s.x);
        }
    };
}

/**
 * 视觉子地图类，用于视觉与lidar点云融合跟踪
*/
class VisualSubMap
{
public:
    VisualSubMap(const std::string &cam_calib_file, const std::string &model_param_str, const std::string &model_file_str);
    ~VisualSubMap();
    // 输入: 当前时间戳，当前图像，gravity(world)系下的点云，当前激光雷达的姿态和位置，LIO中的world系到重力系下的转换
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> addImageCloudTrackFeature(const double _t, const cv::Mat &_img, const PointCloudType::Ptr _cloud_in,
                                   const faster_lio::common::M3D &_r_cur, const faster_lio::common::V3D &_p_cur, const faster_lio::common::M3D &_r_g_w);
    void setLidarCameraExtrinsic(const Eigen::Vector3d &T, const Eigen::Matrix3d &R);
    void setIMULidarExtrinsic(const Eigen::Vector3d &T, const Eigen::Matrix3d &R);
private:
    bool in_border(const Eigen::Vector2d &pt);
    void letnet_init();
    void letnet_convert(const cv::Mat& image_bgr);
    float get_px_val(const cv::Mat &img, float x, float y);
    std::vector<int> sort_and_return_indices(const std::vector<float>& input);
    static bool compare(const std::pair<float,int>& a, const std::pair<float,int>& b);
    // 特征选取
    std::vector<cv::Point3f> feature_extractor(const std::vector<cv::Point2f>& projections,    // 从雷达投影到图像平面的像素点
                                               const std::vector<float>& scores_from_letnet,   // 从神经网络的scoremap得到的像素评分  
                                               const std::vector<float>& depths_form_lidar,    // projections对应的深度
                                               const std::vector<cv::Point2f>& exist_points = std::vector<cv::Point2f>());   // 已有的点，从光流优化过来的

    template < typename T > 
    void reduce_vector( std::vector< T > &v, std::vector< uchar > status )
    {
        int j = 0;
        for ( unsigned int i = 0; i < v.size(); i++ )
            if ( status[ i ] )
                v[ j++ ] = v[ i ];
        v.resize( j );
    }
    
    size_t cnt_ = 0;
    int h_;
    int w_;
    Eigen::Matrix3d R_cl_;
    Eigen::Vector3d t_cl_;
    Eigen::Matrix3d R_il_;
    Eigen::Vector3d t_il_;
    std::string param_path_; 
    camodocal::CameraPtr m_camera_;

    cv::Mat img_gray_;
    float voxel_size_;
    std::set<VOXEL_KEY> voxels_current_;
    unordered_map<VOXEL_KEY,POINTS_IN_VOXEL*> v_sub_map_;

    int letnet_h_;
    int letnet_w_;
    float scale_;
    int score_thd_;
    cv::Mat score_;
    cv::Mat desc_;
    cv::Mat last_desc_;
    const float mean_vals_[3] = {0, 0, 0};
    const float norm_vals_[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    const float mean_vals_inv_[3] = {0, 0, 0};
    const float norm_vals_inv_[3] = {255.f, 255.f, 255.f};
    ncnn::Net net_;
    ncnn::Mat in_;
    ncnn::Mat out1_, out2_;
    std::string model_param_;
    std::string model_file_;
    int max_dist_;
    int max_point_num_;
    std::vector<cv::Point2f> tracked_pixels_;
    std::vector<float> tracked_point_depths_;
    std::vector<cv::Point2f> tracked_pixels_prev_;
    std::vector<size_t> ids_;
    size_t id_cnt_;
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> ln_map_; 
    std::vector<cv::Point2f> ln_pts_velocity
    void show_points_111(const cv::Mat& image, const std::vector<cv::Point2f> &pts, const std::string win_name);
    void show_tracks_111(const cv::Mat& image, const std::vector<cv::Point2f> &pxs_prev, const std::vector<cv::Point2f> &pxs_curr, const std::string win_name);
    double cur_time;
    double prev_time;
    std::vector<cv::Point2f> ptsvelocity(std::vector<int> &ids, std::vector<cv::Point2f> &pxs_prev, const std::vector<cv::Point2f> &pxs_curr)

};

#endif