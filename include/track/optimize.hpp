#pragma once
#include "core/convert.hpp"
#include "core/transform.hpp"
#include "track/frame.hpp"

namespace Track
{

struct Config {
    cv::Mat1f intrinsic;
    int level;
    bool is_chatty;
};

struct Scene {
    // Framex2 + xi
    Scene(
        const Frame& pre_frame,
        const Frame& cur_frame,
        const cv::Mat& xi)
        : pre_gray(pre_frame.m_gray_image),
          pre_depth(pre_frame.m_depth_image),
          cur_gray(cur_frame.m_gray_image),
          cur_depth(cur_frame.m_depth_image),
          warped_image(Transform::warpImage(xi, cur_frame.m_gray_image, cur_frame.m_depth_image, cur_frame.m_intrinsic)),
          xi(xi), intrinsic(cur_frame.m_intrinsic), cols(cur_frame.cols), rows(cur_frame.rows) {}

    // 画像x4 + xi + K
    Scene(
        const cv::Mat& pre_gray,
        const cv::Mat& pre_depth,
        const cv::Mat& cur_gray,
        const cv::Mat& cur_depth,
        const cv::Mat1f& xi,
        const cv::Mat1f& intrinsic)
        : pre_gray(pre_gray),
          pre_depth(pre_depth),
          cur_gray(cur_gray),
          cur_depth(cur_depth),
          warped_image(Transform::warpImage(xi, cur_gray, cur_depth, intrinsic)),
          xi(xi), intrinsic(intrinsic), cols(cur_gray.cols), rows(cur_gray.rows) {}

    const cv::Mat pre_gray;
    const cv::Mat pre_depth;
    const cv::Mat cur_gray;
    const cv::Mat cur_depth;
    const cv::Mat warped_image;
    const cv::Mat1f xi;
    const cv::Mat1f intrinsic;
    const int cols;
    const int rows;
};

struct Outcome {
    Outcome(const cv::Mat1f& xi_update, float residual, int valid_pixels)
        : xi_update(xi_update), residual(residual), valid_pixels(valid_pixels) {}
    const cv::Mat1f xi_update;
    const float residual;
    const int valid_pixels;
};

// xi_updateを計算する
Outcome optimize(const Scene& scene);

}  // namespace Track