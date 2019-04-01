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
    VisualOdometry(Config config) : m_config(config)
    {
        m_tracker = std::make_shared<Track::Tracker>(m_config.track_config);
    }

    // Tracker::track()とMap::estimate()を呼ぶ
    cv::Mat1f odometrize(const cv::Mat& gray_image, const cv::Mat& depth_image)
    {
        Frame frame(gray_image, depth_image);
        cv::Mat1f T = m_tracker->track(frame);
        return T;
    }

private:
    const Config m_config;
    std::shared_ptr<Track::Tracker> m_tracker;
    std::shared_ptr<Map::Mapper> m_mapper;
};
}  // namespace System
