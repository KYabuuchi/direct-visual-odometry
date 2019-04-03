#pragma once
#include "map/mapper.hpp"
#include "system/frame.hpp"
#include "track/tracker.hpp"
#include <memory>

namespace System
{
struct Config {
    Track::Config track_config;
};

class VisualOdometry
{
public:
    VisualOdometry(Config config)
        : m_config(config),
          m_tracker(std::make_shared<Track::Tracker>(m_config.track_config))
    {
    }

    cv::Mat1f odometrize(const cv::Mat1f& gray_image, const cv::Mat1f& depth_image)
    {
        // creating a Frame pair
        std::shared_ptr<Frame> ref_frame = m_history.getRefFrame();
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray_image, depth_image, m_config.track_config.intrinsic, 0);

        // Tracking
        cv::Mat1f relative_xi = m_tracker->track(m_ref_frame, frame);
        frame->update_xi(relative_xi, m_ref_frame);

        // Mapping
        m_mapper->estimate(m_history, frame);

        return math::se3::exp(relative_xi);
    }

    // Only use tracking-mode
    cv::Mat1f odometrizeUsingDepth(const cv::Mat1f& gray_image, const cv::Mat1f& depth_image, const cv::Mat1f& sigma_image)
    {
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray_image, depth_image, sigma_image, m_config.track_config.intrinsic, 0);
        cv::Mat1f relative_xi = m_tracker->track(m_ref_frame, frame);

        frame->update_xi(relative_xi, m_ref_frame);
        m_ref_frame = frame;
        return math::se3::exp(relative_xi);
    }

private:
    const Config m_config;
    std::shared_ptr<Track::Tracker> m_tracker;
    std::shared_ptr<Map::Mapper> m_mapper;
    FrameHistory m_history;

    // Only use tracking-mode
    std::shared_ptr<Frame> m_ref_frame = nullptr;
};
}  // namespace System
