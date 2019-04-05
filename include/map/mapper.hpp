#pragma once
#include "system/frame.hpp"
#include <memory>

namespace Map
{

struct Config {
    const bool is_chatty = true;
    const float minimum_movement = 0.10f;  // [m]
    const float predict_sigma = 0.05f;     // [m]
    const float initial_sigma = 0.50f;     // [m]
};

class Mapper
{
    using Frame = System::Frame;
    using FrameHistory = System::FrameHistory;
    using pFrame = std::shared_ptr<System::Frame>;

public:
    Mapper(const Config& config) : m_config(config) {}

    void estimate(FrameHistory frame_history, pFrame frame)
    {
        if (frame_history.size() == 0) {
            initializeHistory(frame_history, frame);
        } else {


            if (needNewFrame(frame)) {
                propagate(frame_history, frame);
            } else {
                // update(frame_history, frame);
            }
        }
        regularize(frame_history);
    }

    // 十分移動したか否かを判定
    bool needNewFrame(pFrame frame);

    // 分散を適切に設定する
    void initializeHistory(const cv::Mat1f& depth, cv::Mat1f& sigma);
    void initializeHistory(FrameHistory frame_history, pFrame frame)
    {
        initializeHistory(frame->m_depth, frame->m_sigma);
        frame_history.setRefFrame(frame);
    }

    // ref_frameをframeで置き換える
    void propagate(FrameHistory frame_history, pFrame frame)
    {
        // copy
        const pFrame& ref = frame->m_ref_frame;

        propagate(
            ref->m_depth, ref->m_sigma, ref->m_age,
            frame->m_depth, frame->m_sigma, frame->m_age,
            frame->m_relative_xi, frame->m_intrinsic);
        frame_history.setRefFrame(frame);
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


    // 深度・分散を更新
    // void update(FrameHistory frame_history, pFrame frame);

    // 深度を拡散
    void regularize(cv::Mat1f& depth, const cv::Mat1f& sigma);
    void regularize(FrameHistory frame_history)
    {
        pFrame frame = frame_history.getRefFrame();
        regularize(frame->m_depth, frame->m_sigma);
    }

private:
    const Config m_config;
};
}  // namespace Map
