#include "core/loader.hpp"
#include "core/params.hpp"

namespace Core
{
std::string Loader::directorize(std::string file_path)
{
    while (true) {
        if (file_path.empty() or *(file_path.end() - 1) == '/')
            break;
        file_path.erase(file_path.end() - 1);
    }
    return file_path;
}

void Loader::createUndistortMap()
{
    cv::Mat1f K = (cv::Mat1f(3, 3) << 780, 0, 378, 0, 796, 220, 0, 0, 1);
    cv::Mat1f D = (cv::Mat1f(5, 1) << -0.0462, 0.152, -0.00429, 0.0117, -0.0725);

    cv::initUndistortRectifyMap(
        K,
        D,
        cv::Mat(),
        K,
        cv::Size(640, 480),
        CV_32FC1,
        m_rgb_map.at(0),
        m_rgb_map.at(1));
    m_map_initialized = true;
}

bool Loader::getNormalizedUndistortedImages(size_t num, cv::Mat1f& rgb_image)
{
    cv::Mat1f normalized_rgb_image;
    if (not getNormalizedImages(num, normalized_rgb_image))
        return false;
    if (not m_map_initialized)
        createUndistortMap();
    cv::remap(normalized_rgb_image, rgb_image, m_rgb_map.at(0), m_rgb_map.at(1),
        cv::InterpolationFlags::INTER_NEAREST, cv::BorderTypes::BORDER_CONSTANT, math::INVALID);
    return true;
}

bool Loader::getUndistortedImages(size_t num, cv::Mat& rgb_image)
{
    if (not getRawImages(num, rgb_image))
        return false;
    if (not m_map_initialized)
        createUndistortMap();
    cv::remap(rgb_image, rgb_image, m_rgb_map.at(0), m_rgb_map.at(1),
        cv::InterpolationFlags::INTER_NEAREST, cv::BorderTypes::BORDER_CONSTANT, math::INVALID);
    return true;
}

bool Loader::getNormalizedImages(size_t num, cv::Mat1f& normalized_rgb_image)
{
    cv::Mat rgb_image, depth_image;
    if (not getRawImages(num, rgb_image))
        return false;
    cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2GRAY);
    rgb_image.convertTo(normalized_rgb_image, CV_32FC1, 1.0 / 255.0);  // 0~1
    return true;
}
bool Loader::getRawImages(size_t num, cv::Mat& rgb_image)
{
    if (num >= m_rgb_paths.size()) {
        std::cout << "[ERROR] can not open " << num << std::endl;
        return false;
    }
    std::cout << "open " << m_rgb_paths.at(num) << std::endl;
    rgb_image = cv::imread(m_rgb_paths.at(num), cv::IMREAD_UNCHANGED);
    return true;
}

// ============================

void KinectLoader::createUndistortMap()
{
    cv::initUndistortRectifyMap(
        Params::DEPTH().intrinsic, Params::DEPTH().distortion, cv::Mat(),
        Params::DEPTH().intrinsic, Params::DEPTH().resolution, CV_32FC1,
        m_depth_map.at(0), m_depth_map.at(1));
    cv::initUndistortRectifyMap(
        Params::RGB().intrinsic, Params::RGB().distortion, cv::Mat(),
        Params::RGB().intrinsic, Params::RGB().resolution, CV_32FC1,
        m_rgb_map.at(0), m_rgb_map.at(1));
    m_map_initialized = true;
}

bool KinectLoader::getMappedImages(size_t num, cv::Mat1f& mapped_image, cv::Mat1f& depth_image, cv::Mat1f& sigma_image)
{
    cv::Mat1f rgb_image;
    if (not getNormalizedUndistortedImages(num, rgb_image, depth_image))
        return false;

    std::pair<cv::Mat1f, cv::Mat1f> pair = Transform::mapDepthtoGray(depth_image, rgb_image);
    mapped_image = pair.first;
    sigma_image = pair.second;
    return true;
}

bool KinectLoader::getNormalizedUndistortedImages(size_t num, cv::Mat1f& rgb_image, cv::Mat1f& depth_image)
{
    cv::Mat1f normalized_rgb_image;
    cv::Mat1f normalized_depth_image;
    if (not getNormalizedImages(num, normalized_rgb_image, normalized_depth_image))
        return false;

    if (not m_map_initialized)
        createUndistortMap();

    cv::remap(normalized_rgb_image, rgb_image, m_rgb_map.at(0), m_rgb_map.at(1),
        cv::InterpolationFlags::INTER_NEAREST, cv::BorderTypes::BORDER_CONSTANT, math::INVALID);
    cv::remap(normalized_depth_image, depth_image, m_depth_map.at(0), m_depth_map.at(1),
        cv::InterpolationFlags::INTER_NEAREST, cv::BorderTypes::BORDER_CONSTANT, 0);

    return true;
}

bool KinectLoader::getUndistortedImages(size_t num, cv::Mat& rgb_image, cv::Mat& depth_image)
{
    if (not getRawImages(num, rgb_image, depth_image))
        return false;

    if (not m_map_initialized)
        createUndistortMap();

    cv::remap(rgb_image, rgb_image, m_rgb_map.at(0), m_rgb_map.at(1),
        cv::InterpolationFlags::INTER_NEAREST, cv::BorderTypes::BORDER_CONSTANT, math::INVALID);
    cv::remap(depth_image, depth_image, m_depth_map.at(0), m_depth_map.at(1),
        cv::InterpolationFlags::INTER_NEAREST, cv::BorderTypes::BORDER_CONSTANT, 0);

    return true;
}

bool KinectLoader::getNormalizedImages(size_t num, cv::Mat1f& normalized_rgb_image, cv::Mat1f& normalized_depth_image)
{
    cv::Mat rgb_image, depth_image;
    if (not getRawImages(num, rgb_image, depth_image))
        return false;

    cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2GRAY);
    rgb_image.convertTo(normalized_rgb_image, CV_32FC1, 1.0 / 255.0);       // 0~1
    depth_image.convertTo(normalized_depth_image, CV_32FC1, 1.0 / 5000.0);  // [m]
    return true;
}
bool KinectLoader::getRawImages(size_t num, cv::Mat& rgb_image, cv::Mat& depth_image)
{
    if (num >= m_rgb_paths.size()) {
        std::cout << "[ERROR] can not open " << num << std::endl;
        return false;
    }

    std::cout << "open " << m_rgb_paths.at(num) << " " << m_depth_paths.at(num) << std::endl;
    rgb_image = cv::imread(m_rgb_paths.at(num), cv::IMREAD_UNCHANGED);
    depth_image = cv::imread(m_depth_paths.at(num), cv::IMREAD_UNCHANGED);
    return true;
}

}  // namespace Core