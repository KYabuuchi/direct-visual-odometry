#pragma once
#include "map/mapper.hpp"
#include "system/frame.hpp"
#include "track/tracker.hpp"
#include <memory>

// #define SHOW_KEYFRAME

namespace System
{

class VisualOdometry
{
public:
    VisualOdometry(const cv::Mat1f& K) : K(K)
    {
#ifdef SHOW_KEYFRAME
        cv::namedWindow("KeyFrame", cv::WINDOW_NORMAL);
        cv::resizeWindow("KeyFrame", 640, 480);
#endif
    }

    // 初期深度・分散を設定するとき
    VisualOdometry(
        const cv::Mat1f& gray,
        const cv::Mat1f& depth,
        const cv::Mat1f& sigma,
        const cv::Mat1f& K) : K(K)
    {
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray, depth, sigma, K, 4, 1);
        m_history.setRefFrame(frame);
    }

    void showKeyFrame()
    {
        std::vector<cv::Mat> images;
        for (int i = 0, N = m_history.size(); i < N; i++) {
            std::shared_ptr<Frame> frame = m_history[i];
            images.push_back(frame->gray());
        }
        Draw::showImage("KeyFrame", images);
    }

    cv::Mat1f odometrize(const cv::Mat1f& gray_image)
    {
        // create Frame pair
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray_image, K, 2, 2);
        std::shared_ptr<Frame> ref_frame = m_history.getRefFrame();
        if (ref_frame == nullptr) {
            std::cout << "culled K\n"
                      << frame->K() << std::endl;
            m_history.setRefFrame(frame);
            return math::se3::T();
        }

        // Tracking
        cv::Mat1f relative_xi = m_tracker.track(frame, ref_frame);
        frame->updateXi(relative_xi, ref_frame);
        std::cout << "\nT_w:\n"
                  << math::se3::exp(frame->m_xi)
                  << "\nT_r:\n"
                  << math::se3::exp(frame->m_relative_xi)
                  << "\nT_f:\n"
                  << math::se3::exp(frame->m_ref_frame->m_xi) << std::endl;

        // Mapping
        m_mapper.estimate(m_history, frame);

#ifdef SHOW_KEYFRAME
        showKeyFrame();
#endif
        std::cout << std::endl;
        return math::se3::exp(frame->m_xi);
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
        std::cout << std::endl;
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
