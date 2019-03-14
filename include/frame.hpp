#pragma once
#include "converter.hpp"
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

// class Scene
// {
// public:
//     Scene(std::vector<Frame> pre_frames, std::vector<Frame> cur_frames)
//         : m_pre_frames(pre_frames), m_cur_frames(cur_frames) {}

//     Scene(std::vector<Frame> pre_frames, std::vector<Frame> cur_frames)
//         : m_pre_frames(pre_frames), m_cur_frames(cur_frames) {}
//     std::vector<Frame> m_pre_frames;
//     std::vector<Frame> m_cur_frames;
// };

Frame downscaleFrame(const Frame& frame)
{
    cv::Mat depth_image = Converter::cullImage(frame.m_depth_image);
    cv::Mat gray_image = Converter::cullImage(frame.m_gray_image);
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

    // TODO:
    std::reverse(frames.begin(), frames.end());
    return frames;
}
