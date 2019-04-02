#pragma once
#include "core/convert.hpp"
#include "core/math.hpp"
#include <memory>

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
        : id(++latest_id), cols(gray_image.cols), rows(gray_image.rows),
          m_gray(Convert::cullImage(gray_image, times)),
          m_depth(Convert::cullImage(depth_image, times)),
          m_sigma(cv::Mat::ones(gray_image.size(), CV_32FC1) * INITIAL_SIGMA),
          m_age(cv::Mat::zeros(gray_image.size(), CV_32FC1)),
          m_intrinsic(intrinsic),
          m_xi(math::se3::xi()), m_relative_xi(math::se3::xi()), m_ref_frame(nullptr) {}

    // copy constructor
    Frame(const Frame& frame)
        : id(frame.id), cols(frame.cols), rows(frame.rows),
          m_gray(frame.m_gray),
          m_depth(frame.m_depth),
          m_sigma(frame.m_sigma),
          m_age(frame.m_age),
          m_intrinsic(frame.m_intrinsic),
          m_xi(frame.m_xi), m_relative_xi(frame.m_relative_xi), m_ref_frame(frame.m_ref_frame) {}

    // Tracker::trackの後に呼ばれる
    void update_xi(cv::Mat1f relative_xi, std::shared_ptr<Frame> ref_frame)
    {
        if (ref_frame == nullptr)
            return;
        m_relative_xi = relative_xi;
        m_ref_frame = ref_frame;
        m_xi = math::se3::concatenate(ref_frame->m_xi, relative_xi);
    }

    const int id;
    const int cols;
    const int rows;
    static int latest_id;

    // TODO: 深度->逆深度
    cv::Mat1f m_gray;
    cv::Mat1f m_depth;
    cv::Mat1f m_sigma;
    cv::Mat1f m_age;
    cv::Mat1f m_intrinsic;

    // 姿勢
    cv::Mat1f m_xi;
    cv::Mat1f m_relative_xi;
    std::shared_ptr<Frame> m_ref_frame;
};

class FrameHistory
{
public:
    FrameHistory() {}

    void reduceHistory(int remain)
    {
        if (remain < m_history.size())
            m_history.erase(m_history.begin() + remain, m_history.end());
        else
            std::cout << " invalid remains" << std::endl;
    }
    void setRefFrame(std::shared_ptr<Frame> frame) { m_history.push_back(frame); }

    std::shared_ptr<Frame> getRefFrame()
    {
        if (m_history.empty())
            return nullptr;
        return m_history.at(0);
    }

    size_t size() { m_history.size(); }

    std::vector<std::shared_ptr<Frame>> m_history;
};

}  // namespace System