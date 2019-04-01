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
    Frame(const cv::Mat1f& gray_image, const cv::Mat1f& depth_image,
        const cv::Mat1f& intrinsic, const int times)
        : m_gray(Convert::cullImage(gray_image, times)),
          m_depth(Convert::cullImage(depth_image, times)),
          m_sigma(cv::Mat::ones(gray_image.size(), CV_32FC1) * INITIAL_SIGMA),
          m_age(cv::Mat::zeros(gray_image.size(), CV_32FC1)),
          m_intrinsic(intrinsic),
          id(++latest_id), cols(gray_image.cols), rows(gray_image.rows) {}

    // copy constructor
    Frame(const Frame& frame)
        : m_gray(frame.m_gray),
          m_depth(frame.m_depth),
          m_sigma(frame.m_sigma),
          m_age(frame.m_age),
          m_intrinsic(frame.m_intrinsic),
          id(frame.id), cols(frame.cols), rows(frame.rows) {}

    // TODO: 深度->逆深度
    // 輝度・深度・標準偏差・寿命
    cv::Mat1f m_gray;
    cv::Mat1f m_depth;
    cv::Mat1f m_sigma;
    cv::Mat1f m_age;
    cv::Mat1f m_intrinsic;

    const int id;
    const int cols;
    const int rows;
    static int latest_id;

private:
};


}  // namespace System