#pragma once
#include "core/math.hpp"
#include "core/transform.hpp"
#include "system/frame.hpp"
#include <memory>

namespace Map
{
using Frame = System::Frame;
using FrameHistory = System::FrameHistory;
using pFrame = std::shared_ptr<System::Frame>;
using pScene = std::shared_ptr<System::Scene>;

// Gauss分布
struct Gaussian {
    Gaussian(float depth, float sigma) : depth(depth), sigma(sigma) {}
    float depth;
    float sigma;

    void operator()(float d, float s)
    {
        float v1 = math::square(sigma);
        float v2 = math::square(s);
        float v = v1 + v2;

        // 期待値が離れすぎていたら反映しない
        float diff = std::abs(d - depth);
        if (diff > std::max(sigma, s))
            return;

        depth = (v2 * depth + v1 * d) / v;
        sigma = (v1 * v2) / v;
    }
};

// Eipolar線分
struct EpipolarSegment {
    EpipolarSegment(
        const cv::Mat1f& xi,
        const cv::Point2i& x_i,
        const cv::Mat1f& K,
        const float depth,
        const float sigma)
        : min(depth - sigma), max(depth + sigma),
          start(Transform::warp(xi, x_i, max, K)),
          end(Transform::warp(xi, x_i, min, K)),
          length(static_cast<float>(cv::norm(start - end))) {}

    // copy constractor
    EpipolarSegment(const EpipolarSegment& es)
        : min(es.min), max(es.max), start(es.start), end(es.end), length(es.length) {}

    const float min;
    const float max;
    const cv::Point2f start;
    const cv::Point2f end;
    const float length;
};

// Mapperの設定
struct Config {
    Config()
        : is_chatty(true),
          minimum_movement(0.10f),
          predict_sigma(0.10f),
          predict_variance(math::square(predict_sigma)),
          initial_sigma(0.50f),
          initial_variance(math::square(initial_sigma)),
          luminance_sigma(0.00f),
          luminance_variance(math::square(luminance_sigma)),
          epipolar_sigma(0.50f),
          epipolar_variance(math::square(epipolar_sigma)) {}

    // NOTE: 宣言の順番に注意(安易に入れ替えてはいけない)
    const bool is_chatty;
    const float minimum_movement;  // [m]
    const float predict_sigma;     // [m]
    const float predict_variance;
    const float initial_sigma;  // [m]
    const float initial_variance;
    const float luminance_sigma;  // [pixel]
    const float luminance_variance;
    const float epipolar_sigma;  // [pixel]
    const float epipolar_variance;
};

// 本体
class Mapper
{
public:
    Mapper(const Config& config) : m_config(config) {}

    // 本体
    void estimate(FrameHistory& frame_history, pFrame frame);

    // 十分移動したか否かを判定
    bool needNewFrame(pFrame frame);

    // ====Propagate====
    // Ref Frameを伝搬
    pFrame propagate(const FrameHistory& frame_history, const pFrame frame);
    std::tuple<cv::Mat1f, cv::Mat1f, cv::Mat1f> propagate(
        const cv::Mat1f& ref_depth,
        const cv::Mat1f& ref_sigma,
        const cv::Mat1f& ref_age,
        const cv::Mat1f& xi,
        const cv::Mat1f& intrinsic);

    // ====Update====
    // 深度・分散を更新
    void update(const FrameHistory& frame_history, pFrame frame);
    std::tuple<float, float> update(
        const cv::Mat1f& obj_gray,
        const cv::Mat1f& ref_gray,
        const cv::Mat1f& ref_gradx,
        const cv::Mat1f& ref_grady,
        const cv::Mat1f& relative_xi,
        const cv::Mat1f& K,
        const cv::Point2i& x_i,
        float depth,
        float sigma);


    // 該当する画素を探索する
    cv::Point2f doMatching(const cv::Mat1f& ref_gray, const float gray, const EpipolarSegment& es);
    // 深度を推定
    float depthEstimate(
        const cv::Point2f& ref_x_i,
        const cv::Point2f& obj_x_i,
        const cv::Mat1f& K,
        const cv::Mat1f& xi);
    // 分散を推定
    float sigmaEstimate(
        const cv::Mat1f& ref_grad_x,
        const cv::Mat1f& ref_grad_y,
        const cv::Point2f& ref_x_i,
        const EpipolarSegment& es);

    // ====Regularize====
    // 深度を拡散
    void regularize(const FrameHistory& frame_history, pFrame frame);
    void regularize(cv::Mat1f& depth, const cv::Mat1f& sigma);

private:
    const Config m_config;
};
}  // namespace Map
