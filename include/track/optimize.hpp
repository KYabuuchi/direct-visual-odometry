#pragma once
#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/transform.hpp"

namespace Track
{
struct Outcome;
struct Scene;

Outcome optimize(const Scene& scene);

struct Frame {
    Frame(cv::Mat depth_image, cv::Mat gray_image, cv::Mat1f intrinsic)
        : depth(depth_image), gray(gray_image),
          intrinsic(intrinsic),
          cols(depth_image.cols), rows(depth_image.rows) {}
    cv::Mat depth;
    cv::Mat gray;
    cv::Mat1f intrinsic;
    int cols;
    int rows;

    static std::vector<Frame> createFramePyramid(
        const cv::Mat& depth_image,
        const cv::Mat& gray_image,
        const cv::Mat1f& intrinsic,
        const int level)
    {
        std::vector<Frame> frames;
        Frame origin = Frame(depth_image, gray_image, intrinsic);
        for (int i = 0; i < level; i++) {
            frames.push_back(downscaleFrame(origin, level - i - 1));  // level-1 , ... , 1 , 0
        }
        return frames;
    }

    static Frame downscaleFrame(const Frame& frame, int times = 1)
    {
        if (times == 0)
            return Frame(frame);
        cv::Mat depth_image = Convert::cullImage(frame.depth, times);
        cv::Mat gray_image = Convert::cullImage(frame.gray, times);
        cv::Mat1f intrinsic = Convert::cullIntrinsic(frame.intrinsic, times);
        return Frame(depth_image, gray_image, intrinsic);
    }
};

struct Scene {
    Scene(
        const Frame& pre_frame,
        const Frame& cur_frame,
        const cv::Mat1f& xi)
        : pre_gray(pre_frame.gray),
          pre_depth(pre_frame.depth),
          cur_gray(cur_frame.gray),
          cur_depth(cur_frame.depth),
          grad_x(Convert::gradiate(cur_frame.gray, true)),
          grad_y(Convert::gradiate(cur_frame.gray, false)),
          warped_gray(Transform::warpImage(xi, cur_frame.gray, cur_frame.depth, cur_frame.intrinsic)),
          xi(xi), intrinsic(cur_frame.intrinsic),
          cols(cur_frame.cols), rows(cur_frame.rows) {}

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


}  // namespace Track