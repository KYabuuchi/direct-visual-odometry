#pragma once
#include "core/transform.hpp"

namespace Map
{
namespace Update
{
// 輝度画像x2と相対座標から指定画素の深度と分散を計算する
// return: depth,sigma
// return: -1,-1 if failed
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
}  // namespace Update
}  // namespace Map