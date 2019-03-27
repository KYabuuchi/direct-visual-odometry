#include "track/frame.hpp"
#include "core/convert.hpp"
#include "core/math.hpp"

namespace Track
{
Frame downscaleFrame(const Frame& frame, int times)
{
    cv::Mat depth_image = Convert::cullImage(frame.m_depth_image, times);
    cv::Mat gray_image = Convert::cullImage(frame.m_gray_image, times);
    cv::Mat1f intrinsic = frame.m_intrinsic / math::pow(2, times);
    intrinsic(2, 2) = 1;

    return Frame(depth_image, gray_image, intrinsic);
}

std::vector<Frame> createFramePyramid(
    const cv::Mat& depth_image,
    const cv::Mat& gray_image,
    const cv::Mat1f& intrinsic,
    const int level)
{
    std::vector<Frame> frames;
    frames.push_back(Frame(depth_image, gray_image, intrinsic));
    for (int i = 0; i < level - 1; i++)
        frames.push_back(downscaleFrame(frames.at(i)));

    std::reverse(frames.begin(), frames.end());
    return frames;
}

}  // namespace Track