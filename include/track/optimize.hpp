#pragma once
#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/transform.hpp"

namespace Track
{
struct Outcome;
struct Stuff;

Outcome optimize(const Stuff& stuff);

struct Scene {
    Scene(cv::Mat depth_image, cv::Mat gray_image, cv::Mat1f intrinsic)
        : depth(depth_image), gray(gray_image),
          intrinsic(intrinsic),
          cols(depth_image.cols), rows(depth_image.rows) {}
    cv::Mat depth;
    cv::Mat gray;
    cv::Mat1f intrinsic;
    int cols;
    int rows;

    static std::vector<Scene> createFramePyramid(
        const cv::Mat& depth_image,
        const cv::Mat& gray_image,
        const cv::Mat1f& intrinsic,
        const int level)
    {
        std::vector<Scene> frames;
        Scene origin = Scene(depth_image, gray_image, intrinsic);
        for (int i = 0; i < level; i++) {
            frames.push_back(downscaleFrame(origin, level - i - 1));  // level-1 , ... , 1 , 0
        }
        return frames;
    }

    static Scene downscaleFrame(const Scene& frame, int times = 1)
    {
        if (times == 0)
            return Scene(frame);
        cv::Mat depth_image = Convert::cullImage(frame.depth, times);
        cv::Mat gray_image = Convert::cullImage(frame.gray, times);
        cv::Mat1f intrinsic = Convert::cullIntrinsic(frame.intrinsic, times);
        return Scene(depth_image, gray_image, intrinsic);
    }
};

struct Stuff {
    Stuff(
        const Scene& pre,
        const Scene& cur,
        const cv::Mat1f& xi)
        : pre_gray(pre.gray),
          pre_depth(pre.depth),
          cur_gray(cur.gray),
          cur_depth(cur.depth),
          grad_x(Convert::gradiate(cur.gray, true)),
          grad_y(Convert::gradiate(cur.gray, false)),
          warped_gray(Transform::warpImage(xi, cur.gray, cur.depth, cur.intrinsic)),
          xi(xi), intrinsic(cur.intrinsic),
          cols(cur.cols), rows(cur.rows) {}

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