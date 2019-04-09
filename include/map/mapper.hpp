#pragma once
#include "core/math.hpp"
#include "core/transform.hpp"
#include "system/frame.hpp"
#include <memory>

namespace Map
{
using pScene = std::shared_ptr<System::Scene>;

struct Config {
    Config()
        : is_chatty(true),
          minimum_movement(0.10f),
          predict_sigma(0.10f),
          predict_variance(math::square(predict_sigma)),
          initial_sigma(0.50f),
          initial_variance(math::square(initial_sigma)),
          luminance_sigma(1.00f),
          luminance_variance(math::square(luminance_sigma)),
          epipolar_sigma(2.00f),
          epipolar_variance(math::square(epipolar_sigma)) {}

    // NOTE: 宣言の順番に注意
    const bool is_chatty;
    const float minimum_movement = 0.10f;  // [m]
    const float predict_sigma = 0.10f;     // [m]
    const float initial_sigma = 0.50f;     // [m]
    const float luminance_sigma = 1.00f;   // [pixel]
    const float epipolar_sigma = 2.00f;    // [pixel]

    const float predict_variance;
    const float initial_variance;
    const float luminance_variance;
    const float epipolar_variance;
};

class Mapper
{
    struct Gaussian;
    struct EpipolarSegment;

    using Frame = System::Frame;
    using FrameHistory = System::FrameHistory;
    using pFrame = std::shared_ptr<System::Frame>;

public:
    Mapper(const Config& config) : m_config(config) {}

    void estimate(FrameHistory& frame_history, pFrame frame);

    // 十分移動したか否かを判定
    bool needNewFrame(pFrame frame);

    // 分散を適切に設定する
    void initializeHistory(const cv::Mat1f& depth, cv::Mat1f& sigma);
    void initializeHistory(FrameHistory& frame_history, pFrame frame)
    {
        // initializeHistory(frame->m_depth, frame->m_sigma);
        // frame_history.setRefFrame(frame);
    }

    // ====Propagate====
    // ref_frameをframeへ移す
    void propagate(FrameHistory& frame_history, pFrame frame)
    {
        // if (m_config.is_chatty)
        //     std::cout << "propagate" << std::endl;
        // const pFrame& ref = frame->m_ref_frame;
        // propagate(
        //     ref->m_depth, ref->m_sigma, ref->m_age,
        //     frame->m_depth, frame->m_sigma, frame->m_age,
        //     frame->m_relative_xi, frame->m_K);
        // frame_history.setRefFrame(frame);
    }
    void propagate(
        const cv::Mat1f& ref_depth,
        const cv::Mat1f& ref_sigma,
        const cv::Mat1f& ref_age,
        cv::Mat1f& depth,
        cv::Mat1f& sigma,
        cv::Mat1f& age,
        const cv::Mat1f& xi,
        const cv::Mat1f& intrinsic);

    // ====Update====
    // 深度・分散を更新
    void update(FrameHistory& frame_history, pFrame frame);
    // 該当する画素を探索する
    cv::Point2f doMatching(const cv::Mat1f& ref_gray, const float gray, const EpipolarSegment& es);
    // 深度を推定
    float depthEstimate(
        const cv::Mat1f& ref_x_i,
        const cv::Mat1f& obj_x_i,
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
    void regularize(cv::Mat1f& depth, const cv::Mat1f& sigma);
    void regularize(FrameHistory& frame_history)
    {
        // pFrame frame = frame_history.getRefFrame();
        // regularize(frame->m_depth, frame->m_sigma);
    }

private:
    const Config m_config;

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
            const float min,
            const float max)
            : min(min), max(max),
              start(Transform::warp(xi, x_i, max, K)),
              end(Transform::warp(xi, x_i, min, K)),
              length(cv::norm(start - end)) {}

        // copy constractor
        EpipolarSegment(const EpipolarSegment& es)
            : min(es.min), max(es.max), start(es.start), end(es.end), length(es.length) {}

        const float min;
        const float max;
        const cv::Point2f start;
        const cv::Point2f end;
        const float length;
    };
};
}  // namespace Map
