#pragma once
#include "core/transform.hpp"
#include "system/frame.hpp"
#include "track/optimize.hpp"
#include <memory>

namespace Track
{

struct Config {
    const cv::Mat1f intrinsic;
    const int level;
    const bool is_chatty;
    const float minimum_update;
    const float minimum_residual;
};


class Tracker
{
public:
    using pScene = std::shared_ptr<System::Scene>;
    Tracker(const Config& config) : m_config(config), m_initialized(false) {}

    cv::Mat1f track(
        const std::shared_ptr<System::Frame> ref_frame,
        const std::shared_ptr<System::Frame> cur_frame);

private:
    Config m_config;

    bool m_initialized;
};


}  // namespace Track