#pragma once
#include "core/transform.hpp"

namespace Map
{
namespace Implement
{

// return: depth,sigma (-1,-1) if failed
std::tuple<float, float> update(
    const cv::Mat1f& obj_gray,
    const cv::Mat1f& ref_gray,
    const cv::Mat1f& ref_gradx,
    const cv::Mat1f& ref_grady,
    const cv::Mat1f& relative_xi,
    const cv::Mat1f& K,
    const cv::Point2i& x_i,
    float depth,
    float sigma);

std::tuple<cv::Mat1f, cv::Mat1f, cv::Mat1f> propagate(
    const cv::Mat1f& ref_depth,
    const cv::Mat1f& ref_sigma,
    const cv::Mat1f& ref_age,
    const cv::Mat1f& xi,
    const cv::Mat1f& K);

void regularize(cv::Mat1f& depth, const cv::Mat1f& sigma);

}  // namespace Implement
}  // namespace Map