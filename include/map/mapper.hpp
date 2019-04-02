#pragma once
#include "system/frame.hpp"
#include <memory>

namespace Map
{
class Mapper
{
    using Frame = System::Frame;

public:
    Mapper() {}

    void Estimate(
        std::vector<std::shared_ptr<Frame>> /*frame_history*/,
        std::shared_ptr<Frame> /*new_frame*/)
    {
        // bool flag= insertHistory(new_frame);
        // if(flag)
        // propagate(new_frame);
        // else
        // update(frame_history,new_frame);
    }
};
}  // namespace Map
