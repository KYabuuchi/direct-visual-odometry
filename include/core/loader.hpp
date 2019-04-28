#pragma once
#include "calibration/loader.hpp"
#include "core/convert.hpp"
#include "core/transform.hpp"
#include "math/math.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace Core
{
class Loader
{
protected:
    std::array<cv::Mat, 2> m_rgb_map;
    std::vector<std::string> m_rgb_paths;
    virtual void createUndistortMap();

    bool m_map_initialized = false;

    // ファイル名部分を消す
    std::string directorize(std::string file_path);

    Loader() {}

    Calibration::IntrinsicParams RGB;

public:
    Loader(
        const std::string& file_paths_file,
        const std::string& config_file)
    {
        std::ifstream ifs(file_paths_file);
        if (not ifs.is_open()) {
            std::cout << "[ERROR] can not open " << file_paths_file << std::endl;
            abort();
        }
        std::string dir = directorize(file_paths_file);
        while (not ifs.eof()) {
            std::string file_path;
            std::getline(ifs, file_path);
            if (not file_path.empty()) {
                std::istringstream stream(file_path);
                std::string file_path;
                getline(stream, file_path, ' ');
                m_rgb_paths.push_back(dir + file_path);
            }
        }
        ifs.close();

        Calibration::Loader config(config_file);
        RGB = config.monocular();
    }

    const Calibration::IntrinsicParams& Rgb() const { return RGB; }

    // 正規除歪画像 CV_32FC1
    bool getNormalizedUndistortedImages(size_t num, cv::Mat1f& rgb_image);
    // 除歪画像 CV_8UC3
    bool getUndistortedImages(size_t num, cv::Mat& rgb_image);
    // 正規画像 CV_32FC1
    bool getNormalizedImages(size_t num, cv::Mat1f& normalized_rgb_image);
    // 生画像 CV_8UC3
    bool getRawImages(size_t num, cv::Mat& rgb_image);
};

class KinectLoader : public Loader
{
protected:
    std::array<cv::Mat, 2> m_depth_map;
    std::vector<std::string> m_depth_paths;
    void createUndistortMap() override;

    Calibration::IntrinsicParams DEPTH;
    Calibration::ExtrinsicParams EXT;

public:
    KinectLoader(
        const std::string& file_paths_file,
        const std::string& config_file)
    {
        std::ifstream ifs(file_paths_file);
        if (not ifs.is_open()) {
            std::cout << "[ERROR] can not open " << file_paths_file << std::endl;
            abort();
        }
        std::string dir = directorize(file_paths_file);
        while (not ifs.eof()) {
            std::string dual_file_path;
            std::getline(ifs, dual_file_path);
            if (not dual_file_path.empty()) {
                std::istringstream stream(dual_file_path);
                std::string file_path;
                getline(stream, file_path, ' ');
                m_rgb_paths.push_back(dir + file_path);
                getline(stream, file_path, ' ');
                m_depth_paths.push_back(dir + file_path);
            }
        }
        ifs.close();

        Calibration::Loader config(config_file);
        RGB = config.rgb();
        DEPTH = config.depth();
        EXT = config.extrinsic();
    }

    const Calibration::IntrinsicParams& Depth() const { return DEPTH; }
    const Calibration::ExtrinsicParams& Ext() const { return EXT; }

    // 変形除歪画像 CV_32FC1,CV_32FC1
    bool getMappedImages(size_t num, cv::Mat1f& mapped_image, cv::Mat1f& depth_image, cv::Mat1f& sigma_image);
    // 正規除歪画像 CV_32FC1,CV_32FC1
    bool getNormalizedUndistortedImages(size_t num, cv::Mat1f& rgb_image, cv::Mat1f& depth_image);
    // 除歪画像 CV_8UC3,CV_16UC1
    bool getUndistortedImages(size_t num, cv::Mat& rgb_image, cv::Mat& depth_image);
    // 正規画像 CV_32FC1,CV_32FC1
    bool getNormalizedImages(size_t num, cv::Mat1f& normalized_rgb_image, cv::Mat1f& normalized_depth_image);
    // 生画像 CV_8UC3,CV_16UC1
    bool getRawImages(size_t num, cv::Mat& rgb_image, cv::Mat& depth_image);
};
}  // namespace Core