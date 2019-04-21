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
    Tracker() : window_name("Tracking")
    {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, 1280, 720);
    }

    cv::Mat1f track(
        const pFrame obj_frame,
        const pFrame ref_frame);

private:
    const std::string window_name;
};


}  // namespace Track