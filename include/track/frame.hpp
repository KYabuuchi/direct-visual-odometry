#pragma once
#include <opencv2/opencv.hpp>

namespace Track
{
class Frame
{
public:
    Frame(cv::Mat depth_image, cv::Mat gray_image, cv::Mat1f intrinsic)
        : m_depth_image(depth_image), m_gray_image(gray_image),
          m_intrinsic(intrinsic),
          cols(depth_image.cols), rows(depth_image.rows) {}

    cv::Mat m_depth_image;
    cv::Mat m_gray_image;
    cv::Mat1f m_intrinsic;
    int cols;
    int rows;
};

Frame downscaleFrame(const Frame& frame, int times = 1);

std::vector<Frame> createFramePyramid(
    const cv::Mat& depth_image,
    const cv::Mat& gray_image,
    const cv::Mat1f& intrinsic,
    const int level);

}  // namespace Track