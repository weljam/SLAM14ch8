#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

//相机内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
//基线
double baseline = 0.573;
//图像路径
string left_file = "../left.png";
string disparity_file = "../disparity.png";
boost::format fmt_others("../%06d.png");    // other files

//定义后面常用的数据
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// 定义雅可比矩阵类，用于并行计算
class JacobianAccumulator {
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const vector<double> depth_ref_,
        Sophus::SE3d &T21_) :
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /// 雅可比矩阵计算的范围，即地图点的个数
    void accumulate_jacobian(const cv::Range &range);

    /// 获取海森矩阵
    Matrix6d hessian() const { return H; }

    /// 获取误差
    Vector6d bias() const { return b; }

    /// 获取损失
    double cost_func() const { return cost; }

    /// 获取第二帧图像中的像素点
    VecVector2d projected_points() const { return projection; }

    ///重置海森矩阵，误差，以及损失
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }
//私有成员，仅在成员函数中可用
private:
    const cv::Mat &img1;//图像1
    const cv::Mat &img2;//图像2
    const VecVector2d &px_ref;//图像1的像素点
    const vector<double> depth_ref;//图像1的深度，与像素点对应，可用于后续地图点的计算
    Sophus::SE3d &T21;//变换矩阵
    VecVector2d projection; // 图像2中的像素点

    std::mutex hessian_mutex; //上锁
    Matrix6d H = Matrix6d::Zero();//海森矩阵
    Vector6d b = Vector6d::Zero();//误差
    double cost = 0;
};

//多层直接法位姿估计，实际上是在多层金字塔中调用单层直接法
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,//图像1
    const cv::Mat &img2,//图像2
    const VecVector2d &px_ref,//图像1像素点
    const vector<double> depth_ref,//图像1像素点对应的深度
    Sophus::SE3d &T21//变换矩阵
);

//单层直接法位姿估计
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,//图像1
    const cv::Mat &img2,//图像2
    const VecVector2d &px_ref,//图像1的像素点
    const vector<double> depth_ref,//图像1像素点对应的深度
    Sophus::SE3d &T21//变换矩阵
);

// bilinear interpolation双线性插值得到灰度值，参考光流处
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    //边界检查
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    //这里和光流的区别在于这里直接使用灰度值的指针
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +///img.step即一行的偏移量，得到下一行的灰度值
        xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv) {
    //读取图像数据
    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // 随机取第一帧图像的下像素点，并且获取其地图点
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // 产生像素点以及对应的深度
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // 深度计算公式，slam14讲z = fb/d 焦距*基线/视差
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // 评估1-5.png图像的位姿变换
    //定义变换矩阵,自动初始化为【1，0，0，0；0，1，0，0；0，0，1，0；0，0，0，1】单位阵
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        cout<<"第"<<i<<"副图：*****"<<endl;
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        //DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
    return 0;
}

//单层直接法位姿估计
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,//图像1
    const cv::Mat &img2,//图像2
    const VecVector2d &px_ref,//图像1像素点
    const vector<double> depth_ref,//图像1像素点对应的深度
    Sophus::SE3d &T21) {

    const int iterations = 10;//迭代次数
    double cost = 0, lastCost = 0;//损失和总损失
    auto t1 = chrono::steady_clock::now();//计时
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);//实例化雅可比类

    //
    for (int iter = 0; iter < iterations; iter++) {
        //将实例化类重置，计算
        jaco_accu.reset();
        //并行计算0-px_ref.size()对应的像素点
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        //获得H,b矩阵
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        //解出变换矩阵增量，利用李代数计算新变换矩阵
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;
        //获取损失
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {//代价不再减小，则停止迭代
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {//更新量很小了，就停止迭代
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // 画出图像2中像素点的变化
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show,cv::COLOR_GRAY2BGR);
    //获取图像2中的像素点
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        //获取图像1和图像2的像素点
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}

//雅可比计算函数，重点！！
void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {

    // 设置参数，
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++) {

        //计算图像2中的像素点
        //首先计算图像1像素的对应的地图点
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        //计算图像2下的地图点坐标
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0)   //判断在图像2中地图点是否有效
            continue;
        
        //计算对应地图点在图像2中的像素坐标
        float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
        
        //判断像素点是否有效，无效则下一个循环
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        //将图像2的像素点保存
        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;

        //匹配上了就+1
        cnt_good++;

        //计算误差以及雅可比矩阵
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {
                //利用像素点局部灰度值不变计算灰度差，进而迭代优化
                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                
                Matrix26d J_pixel_xi;//u对变换矩阵T李代数的偏导
                Eigen::Vector2d J_img_pixel;//图像2灰度值对像素点的偏导

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                //总雅可比矩阵
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                //计算H，b矩阵，以及损失
                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good) {
        //并行运算，上锁,避免写冲突
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    //金字塔层数
    int pyramids = 4;
    //金字塔缩放比例
    double pyramid_scale = 0.5;
    //金字塔的每层的比例
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    //创建金字塔
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }
        //在不同层，对应的相机参数也改变了
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        //调用单层直接法
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}
