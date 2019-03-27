#pragma once
#include "core/transform.hpp"
#include "track/frame.hpp"
#include "track/optimize.hpp"

namespace Track
{

class Tracker
{
public:
    void plot(bool block = false);

    Tracker(const Config& config) : m_initialized(false), m_config(config)
    {
        cv::namedWindow("show", cv::WINDOW_NORMAL);
    }

    void init(cv::Mat depth_image, cv::Mat gray_image);

    // 相対姿勢を計算する(おまかせ)
    cv::Mat1f track(const cv::Mat& depth_image, const cv::Mat& gray_image);


private:
    void showImage(const Scene& scene);

    bool m_initialized;

    Config m_config;
    std::vector<Frame> m_pre_frames;
    std::vector<Frame> m_cur_frames;
    std::vector<std::vector<float>> m_vector_of_residuals;
};

}  // namespace Track