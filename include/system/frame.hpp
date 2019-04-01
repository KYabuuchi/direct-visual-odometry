#pragma once
#include "core/convert.hpp"


namespace System
{
namespace
{
const float INITIAL_SIGMA = 100.0f;
}

class Frame
{
public:
    // NOTE: sigmaとageは適当
    Frame(const cv::Mat1f gray_image, const cv::Mat1f depth_image)
        : m_gray(Convert::cullImage(gray_image, 1)),
          m_depth(Convert::cullImage(depth_image, 1)),
          m_sigma(cv::Mat::ones(gray_image.size(), CV_32FC1) * INITIAL_SIGMA),
          m_age(cv::Mat::zeros(gray_image.size(), CV_32FC1)),
          m_id(++latest_id) {}

    // copy
    Frame(const Frame& frame)
        : m_gray(frame.m_gray),
          m_depth(frame.m_depth),
          m_sigma(frame.m_sigma),
          m_age(frame.m_age),
          m_id(frame.m_id) {}

    // TODO: 深度->逆深度
    // 輝度・深度・標準偏差・寿命
    cv::Mat1f m_gray;
    cv::Mat1f m_depth;
    cv::Mat1f m_sigma;
    cv::Mat1f m_age;

    const int m_id;
    static int latest_id;

private:
};


}  // namespace System