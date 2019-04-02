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

    void Estimate(FrameHistory frame_history, pFrame new_frame)
    {
        if (frame_history.size() == 0) {
            initializeHistory(frame_history, new_frame);
        } else if (insertableHistory(new_frame)) {
            propagate(frame_history, new_frame);
        } else {
            update(frame_history, new_frame);
        }

        regularize(frame_history);
    }

    bool insertableHistory(pFrame new_frame);
    void initializeHistory(FrameHistory frame_history, pFrame new_frame);
    void propagate(FrameHistory frame_history, pFrame new_frame);
    void update(FrameHistory frame_history, pFrame new_frame);
    void regularize(FrameHistory frame_history);
};
}  // namespace Map
