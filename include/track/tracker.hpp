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
    Tracker(const Config& config) : m_config(config), m_initialized(false) {}

    cv::Mat1f track(
        const std::shared_ptr<System::Frame> ref_frame,
        const std::shared_ptr<System::Frame> cur_frame);

    // NOTE: use only in tracking test
    cv::Mat1f track(const cv::Mat& gray_image, const cv::Mat& depth_image);

private:
    Config m_config;

    // cache
    std::vector<Scene> m_pre_scenes;

    // NOTE: use only in tracking mode
    bool m_initialized;
};


}  // namespace Track