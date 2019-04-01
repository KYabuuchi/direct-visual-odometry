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
          m_tracker(std::make_shared<Track::Tracker>(m_config.track_config)),
          m_ref_frame(nullptr)
    {
    }


    // Depthが既知
    cv::Mat1f odometrizeUsingDepth(const cv::Mat& gray_image, const cv::Mat& depth_image)
    {
        std::cout << "odometry" << std::endl;
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray_image, depth_image, m_config.track_config.intrinsic, 0);
        cv::Mat1f relative_xi = m_tracker->track(m_ref_frame, frame);

        std::cout << "track done" << std::endl;
        frame->update_xi(relative_xi, m_ref_frame);
        m_ref_frame = frame;
        return math::se3::exp(relative_xi);
    }

private:
    const Config m_config;
    std::shared_ptr<Track::Tracker> m_tracker;
    std::shared_ptr<Map::Mapper> m_mapper;
    std::shared_ptr<Frame> m_ref_frame;
};
}  // namespace System
