#pragma once
#include "core/transform.hpp"
#include "system/frame.hpp"
#include "track/optimize.hpp"
#include <memory>

namespace Track
{
class Tracker
{
public:
    using pScene = std::shared_ptr<System::Scene>;
    using pFrame = std::shared_ptr<System::Frame>;
    Tracker() {}

    cv::Mat1f track(
        const pFrame obj_frame,
        const pFrame ref_frame);

private:
};


}  // namespace Track