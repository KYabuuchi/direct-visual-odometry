#pragma once
#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/transform.hpp"
#include "system/frame.hpp"
#include <memory>

namespace Track
{
struct Stuff {
    Stuff(
        std::shared_ptr<System::Scene> obj,
        std::shared_ptr<System::Scene> ref,
        const cv::Mat1f& xi,
        const int levels)
        : obj_gray(obj->gray()),
          ref_gray(ref->gray()),
          ref_depth(ref->depth()),
          ref_sigma(ref->sigma()),
          grad_x(ref->gradX()),
          grad_y(ref->gradY()),
          warped_gray(Transform::warpImage(xi, ref->gray(), ref->depth(), ref->K())),
          xi(xi), K(ref->K()),
          cols(ref->cols), rows(ref->rows), levels(levels) {}

    void update(const cv::Mat1f _xi)
    {
        xi = _xi;
        warped_gray = Transform::warpImage(xi, ref_gray, ref_depth, K);
    }

    void show(const std::string& window_name) const
    {
        Draw::showImage(window_name,
            Draw::visualizeGray(warped_gray),
            Draw::visualizeGray(obj_gray),
            // Draw::visualizeGradient(grad_x),
            // Draw::visualizeGradient(grad_y),
            Draw::visualizeGray(ref_gray),
            Draw::visualizeDepth(ref_depth, ref_sigma));
    }

    const cv::Mat1f obj_gray;
    const cv::Mat1f ref_gray;
    const cv::Mat1f ref_depth;
    const cv::Mat1f ref_sigma;
    const cv::Mat1f grad_x;
    const cv::Mat1f grad_y;
    cv::Mat1f warped_gray;
    cv::Mat1f xi;
    const cv::Mat1f K;
    const int cols;
    const int rows;
    const int levels;
};

struct Outcome {
    Outcome(const cv::Mat1f& xi_update, float residual, int valid_pixels)
        : xi_update(xi_update), residual(residual), valid_pixels(valid_pixels) {}
    const cv::Mat1f xi_update;
    const float residual;
    const int valid_pixels;
};

Outcome optimize(const Stuff& stuff);

}  // namespace Track