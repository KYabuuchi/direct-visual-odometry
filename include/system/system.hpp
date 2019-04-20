#pragma once
#include "map/mapper.hpp"
#include "system/frame.hpp"
#include "track/tracker.hpp"
#include <memory>

namespace System
{

class VisualOdometry
{
public:
    VisualOdometry(const cv::Mat1f& K) : K(K) {}

    VisualOdometry(
        const cv::Mat1f& gray,
        const cv::Mat1f& depth,
        const cv::Mat1f& sigma,
        const cv::Mat1f& K) : K(K)
    {
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray, depth, sigma, K, 4, 2);
        m_history.setRefFrame(frame);
    }

    cv::Mat1f odometrize(const cv::Mat1f& gray_image)
    {
        // create Frame pair
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray_image, K, 4, 2);
        std::shared_ptr<Frame> ref_frame = m_history.getRefFrame();
        if (ref_frame == nullptr) {
            m_history.setRefFrame(frame);
            return math::se3::T();
        }

        // Tracking
        cv::Mat1f relative_xi = m_tracker.track(frame, ref_frame);
        frame->updateXi(relative_xi, ref_frame);
        std::cout << "xi_w: " << frame->m_xi.t() << " xi_r: " << frame->m_relative_xi.t() << std::endl;

        // Mapping
        m_mapper.estimate(m_history, frame);

        return math::se3::exp(relative_xi);
    }

    // NOTE:Only use tracking-mode
    cv::Mat1f odometrizeUsingDepth(
        const cv::Mat1f& gray,
        const cv::Mat1f& depth,
        const cv::Mat1f& sigma)
    {
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray, depth, sigma, K, 4, 1);
        if (m_ref_frame == nullptr) {
            m_ref_frame = frame;
            return math::se3::T();
        }

        cv::Mat1f relative_xi = m_tracker.track(frame, m_ref_frame);
        frame->updateXi(relative_xi, m_ref_frame);
        m_ref_frame = frame;
        return math::se3::exp(relative_xi);
    }

private:
    const cv::Mat1f K;

    Track::Tracker m_tracker;
    Map::Mapper m_mapper;
    FrameHistory m_history;

    // NOTE:Only use tracking-mode
    std::shared_ptr<Frame> m_ref_frame = nullptr;
};
}  // namespace System
