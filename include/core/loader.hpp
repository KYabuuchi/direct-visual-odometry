#pragma once
#include "calibration/params_struct.hpp"
#include "core/params.hpp"
#include "core/transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

class Loader
{
private:
    std::vector<std::string> m_file_paths1;
    std::vector<std::string> m_file_paths2;
    const std::string m_file_paths_file;

    bool m_map_initialized;
    std::array<cv::Mat, 2> m_rgb_map;
    std::array<cv::Mat, 2> m_depth_map;

    // ファイル名部分を消す
    std::string directorize(std::string file_path)
    {
        while (true) {
            if (file_path.empty() or *(file_path.end() - 1) == '/')
                break;
            file_path.erase(file_path.end() - 1);
        }
        return file_path;
    }

    void createUndistortionMap()
    {
        cv::initUndistortRectifyMap(Params::DEPTH().intrinsic, Params::DEPTH().distortion, cv::Mat(),
            Params::DEPTH().intrinsic, Params::DEPTH().resolution, CV_32FC1, m_depth_map.at(0), m_depth_map.at(1));
        cv::initUndistortRectifyMap(Params::RGB().intrinsic, Params::RGB().distortion, cv::Mat(),
            Params::RGB().intrinsic, Params::RGB().resolution, CV_32FC1, m_rgb_map.at(0), m_rgb_map.at(1));

        m_map_initialized = true;
    }


public:
    Loader(const std::string& file_paths_file) : m_file_paths_file(file_paths_file), m_map_initialized(false)
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
                m_file_paths1.push_back(dir + file_path);
                getline(stream, file_path, ' ');
                m_file_paths2.push_back(dir + file_path);
            }
        }
        ifs.close();
    }

    // 変形除歪画像を取得
    bool getMappedImages(size_t num, cv::Mat& mapped_image, cv::Mat& depth_image)
    {
        cv::Mat rgb_image;
        if (not getNormalizedUndistortedImages(num, rgb_image, depth_image))
            return false;

        mapped_image = Transform::mapDepthtoGray(depth_image, rgb_image);
        return true;
    }

    // 変形画像を取得(歪あり)
    bool getMappedDistortedImages(size_t num, cv::Mat& mapped_image, cv::Mat& depth_image)
    {
        cv::Mat rgb_image;
        if (not getNormalizedImages(num, rgb_image, depth_image))
            return false;

        mapped_image = Transform::mapDepthtoGray(depth_image, rgb_image);
        return true;
    }

    // 正規除歪画像を取得
    bool getNormalizedUndistortedImages(size_t num, cv::Mat& rgb_image, cv::Mat& depth_image)
    {
        if (not getUndistortedImages(num, rgb_image, depth_image))
            return false;

        cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2GRAY);
        rgb_image.convertTo(rgb_image, CV_32FC1, 1.0 / 255.0);       // 0~1
        depth_image.convertTo(depth_image, CV_32FC1, 1.0 / 5000.0);  // [m]
        return true;
    }

    // 正規画像を取得
    bool getNormalizedImages(size_t num, cv::Mat& rgb_image, cv::Mat& depth_image)
    {
        if (not getRowImages(num, rgb_image, depth_image))
            return false;

        cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2GRAY);
        rgb_image.convertTo(rgb_image, CV_32FC1, 1.0 / 255.0);       // 0~1
        depth_image.convertTo(depth_image, CV_32FC1, 1.0 / 5000.0);  // [m]
        return true;
    }


    // 除歪画像を取得
    bool getUndistortedImages(size_t num, cv::Mat& rgb_image, cv::Mat& depth_image)
    {
        if (not getRowImages(num, rgb_image, depth_image))
            return false;

        if (not m_map_initialized)
            createUndistortionMap();

        cv::Mat undistorted_rgb_image;
        cv::Mat undistorted_depth_image;
        cv::remap(rgb_image, undistorted_rgb_image, m_rgb_map.at(0), m_rgb_map.at(1), 0);
        cv::remap(depth_image, undistorted_depth_image, m_depth_map.at(0), m_depth_map.at(1), 0);

        rgb_image = undistorted_rgb_image;
        depth_image = undistorted_depth_image;
        return true;
    }

    // 生画像を取得
    bool getRowImages(size_t num, cv::Mat& rgb_image, cv::Mat& depth_image)
    {
        if (num >= m_file_paths1.size()) {
            std::cout << "[ERROR] can not open " << num << std::endl;
            return false;
        }

        std::cout << "open " << m_file_paths1.at(num) << " " << m_file_paths2.at(num) << std::endl;
        rgb_image = cv::imread(m_file_paths1.at(num), cv::IMREAD_UNCHANGED);
        depth_image = cv::imread(m_file_paths2.at(num), cv::IMREAD_UNCHANGED);
        return true;
    }
};