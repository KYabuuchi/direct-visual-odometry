#pragma once
#include "core/transform.hpp"
#include "math/math.hpp"
#include "system/frame.hpp"
#include <memory>

namespace Map
{
using Frame = System::Frame;
using FrameHistory = System::FrameHistory;
using pFrame = std::shared_ptr<System::Frame>;
using pScene = std::shared_ptr<System::Scene>;

// Mapperの設定
struct Config {
    Config()
        : is_chatty(true),
          minimum_movement(0.10f),
          predict_sigma(0.10f),
          predict_variance(math::square(predict_sigma)),
          initial_sigma(0.50f),
          initial_variance(math::square(initial_sigma)),
          luminance_sigma(0.01f),
          luminance_variance(math::square(luminance_sigma)),
          epipolar_sigma(0.5f),
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

    // ====Regularize====
    // 深度を拡散
    void regularize(const FrameHistory& frame_history, pFrame frame);
    void regularize(cv::Mat1f& depth, const cv::Mat1f& sigma);

private:
    const Config m_config;
};
}  // namespace Map
