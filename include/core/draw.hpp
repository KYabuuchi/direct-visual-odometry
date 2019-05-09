#pragma once
#include <opencv2/opencv.hpp>

namespace Draw
{
// NOTE:無効画素は赤色

// 輝度
cv::Mat visualizeGray(const cv::Mat1f& src_image);

// 深度
cv::Mat visualizeDepth(const cv::Mat1f& src_image);
cv::Mat visualizeDepth(const cv::Mat1f& depth, const cv::Mat1f& sigma);
cv::Mat visualizeDepthRaw(const cv::Mat1f& src_image);

// 勾配
cv::Mat visualizeGradient(const cv::Mat1f& src_image);

// 標準偏差
cv::Mat visualizeSigma(const cv::Mat1f& src_image);

// 年齢
cv::Mat visualizeAge(const cv::Mat1f& src_image);

template <class T>
cv::Mat merge(const T& img)
{
    cv::Mat zero = cv::Mat::zeros(img.size(), img.type());
    cv::Mat m;
    cv::vconcat(img, zero, m);
    return m;
}

template <class T>
cv::Mat merge(const T& img1, const T& img2)
{
    cv::Mat m;
    cv::vconcat(img1, img2, m);
    return m;
}

template <class U, class... T>
cv::Mat merge(const U& img1, const U& img2, const T&... tail)
{
    cv::Mat m1;
    cv::vconcat(img1, img2, m1);
    cv::Mat m2 = merge(tail...);
    cv::Mat m3;
    cv::hconcat(m1, m2, m3);
    return m3;
}

template <class... T>
void showImage(const std::string& window_name, const T&... tail)
{
    cv::Mat show_image = merge(tail...);
    cv::imshow(window_name, show_image);
}

cv::Mat merge(std::vector<cv::Mat>& tail);

void showImage(const std::string& window_name, std::vector<cv::Mat>& tail);


}  // namespace Draw
