#pragma once
#include "system/frame.hpp"
#include <memory>

namespace Map
{

struct Config {
    const bool is_chatty;
    const float minimum_movement;
};

class Mapper
{
    using Frame = System::Frame;
    using FrameHistory = System::FrameHistory;
    using pFrame = std::shared_ptr<System::Frame>;

public:
    Mapper() {}

    void estimate(FrameHistory frame_history, pFrame frame)
    {
        if (frame_history.size() == 0) {
            initializeHistory(frame_history, frame);
        } else if (insertableHistory(frame)) {
            propagate(frame_history, frame);
        } else {
            update(frame_history, frame);
        }

        regularize(frame_history);
    }

    bool insertableHistory(pFrame frame);

    // 分散を適切に設定する
    void initializeHistory(const cv::Mat1f& depth, cv::Mat1f& sigma);
    void initializeHistory(FrameHistory frame_history, pFrame frame)
    {
        initializeHistory(frame->m_depth, frame->m_sigma);
        frame_history.setRefFrame(frame);
    }

    // ref_frameをframeで置き換える
    void propagate(FrameHistory frame_history, pFrame frame);

    // 深度・分散を更新
    void update(FrameHistory frame_history, pFrame frame);

    // 深度を拡散
    void regularize(cv::Mat1f& depth, const cv::Mat1f& sigma);
    void regularize(FrameHistory frame_history)
    {
        pFrame frame = frame_history.getRefFrame();
        regularize(frame->m_depth, frame->m_sigma);
    }
};
}  // namespace Map
