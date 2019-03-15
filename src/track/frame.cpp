#include "track/frame.hpp"
#include "core/convert.hpp"

Frame downscaleFrame(const Frame& frame)
{
    cv::Mat depth_image = Convert::cullImage(frame.m_depth_image);
    cv::Mat gray_image = Convert::cullImage(frame.m_gray_image);
    cv::Mat1f intrinsic = frame.m_intrinsic / 2;
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

    // TODO: ほんまか?
    std::reverse(frames.begin(), frames.end());
    return frames;
}
