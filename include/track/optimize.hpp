#pragma once
#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/transform.hpp"
#include "track/frame.hpp"

namespace Track
{
struct Scene {
    // Framex2 + xi
    Scene(
        const Frame& pre_frame,
        const Frame& cur_frame,
        const cv::Mat1f& xi)
        : pre_gray(pre_frame.m_gray_image),
          pre_depth(pre_frame.m_depth_image),
          cur_gray(cur_frame.m_gray_image),
          cur_depth(cur_frame.m_depth_image),
          grad_x(Convert::gradiate(cur_frame.m_gray_image, true)),
          grad_y(Convert::gradiate(cur_frame.m_gray_image, false)),
          warped_gray(Transform::warpImage(xi, cur_frame.m_gray_image,
              cur_frame.m_depth_image, cur_frame.m_intrinsic)),
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
          grad_x(Convert::gradiate(cur_gray, true)),
          grad_y(Convert::gradiate(cur_gray, false)),
          warped_gray(Transform::warpImage(xi, cur_gray, cur_depth, intrinsic)),
          xi(xi), intrinsic(intrinsic),
          cols(cur_gray.cols), rows(cur_gray.rows) {}

    void update(const cv::Mat1f _xi)
    {
        xi = _xi;
        warped_gray = Transform::warpImage(xi, cur_gray, cur_depth, intrinsic);
    }

    void show(const std::string& window_name)
    {
        Draw::showImage(window_name, pre_gray, pre_depth, warped_gray, cur_gray, cur_depth);
    }

    const cv::Mat pre_gray;
    const cv::Mat pre_depth;
    const cv::Mat cur_gray;
    const cv::Mat cur_depth;
    const cv::Mat grad_x;
    const cv::Mat grad_y;
    cv::Mat warped_gray;
    cv::Mat1f xi;
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