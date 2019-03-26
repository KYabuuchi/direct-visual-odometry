#pragma once
#include <opencv2/opencv.hpp>

class Frame
{
public:
    Frame(cv::Mat depth_image, cv::Mat gray_image, cv::Mat1f intrinsic)
        : m_depth_image(depth_image), m_gray_image(gray_image),
          m_intrinsic(intrinsic),
          m_cols(depth_image.cols), m_rows(depth_image.rows) {}

    cv::Mat m_depth_image;
    cv::Mat m_gray_image;
    cv::Mat1f m_intrinsic;
    int m_cols;
    int m_rows;
};

Frame downscaleFrame(const Frame& frame);

std::vector<Frame> createFramePyramid(
    const cv::Mat& depth_image,
    const cv::Mat& gray_image,
    const cv::Mat1f& intrinsic,
    const int level);