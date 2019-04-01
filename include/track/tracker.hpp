#pragma once
#include "core/transform.hpp"
#include "system/frame.hpp"
#include "track/optimize.hpp"

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
    Tracker(const Config& config) : m_initialized(false), m_config(config) {}

    // T(4x4)[m]
    cv::Mat1f track(const cv::Mat& gray_image, const cv::Mat& depth_image);

    // T(4x4)[m]
    cv::Mat1f track(const System::Frame& frame)
    {
        // TODO: sigmaの考慮
        return track(frame.m_gray, frame.m_depth);
    }

private:
    bool m_initialized;

    Config m_config;
    std::vector<Scene> m_pre_scene;
    std::vector<Scene> m_cur_scene;
};


}  // namespace Track