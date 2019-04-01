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
          m_pre_frame(nullptr)
    {
    }

    // Tracker::track()とMap::estimate()を呼ぶ
    cv::Mat1f odometrize(const cv::Mat& gray_image, const cv::Mat& depth_image)
    {
        std::shared_ptr<Frame> frame = std::make_shared<Frame>(gray_image, depth_image, m_config.track_config.intrinsic, 0);
        cv::Mat1f T = m_tracker->track(m_pre_frame, frame);
        m_pre_frame = frame;
        return T;
    }

private:
    const Config m_config;
    std::shared_ptr<Track::Tracker> m_tracker;
    std::shared_ptr<Map::Mapper> m_mapper;
    std::shared_ptr<Frame> m_pre_frame;
};
}  // namespace System
