#include "core/draw.hpp"
#include "core/convert.hpp"
#include "core/math.hpp"

namespace Draw
{
// 無効画素は赤色へ

cv::Mat visualizeGray(const cv::Mat1f& src_image)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, 255);            // change value-depth
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);  // change number of channel
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int* pt) -> void {
            if (math::isValid(src_image(pt[0], pt[1])))
                return;
            p[0] = 255;
        });
    return dst_image;
}

cv::Mat visualizeDepth(const cv::Mat1f& src_image)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, 100);
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);
    cv::cvtColor(dst_image, dst_image, cv::COLOR_BGR2HSV);
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int*) -> void {
            p[0] = static_cast<unsigned char>(p[2] * 90.0 / 255);  // H in [0,179]
            p[1] = 255;
            p[2] = 255;
        });
    cv::cvtColor(dst_image, dst_image, cv::COLOR_HSV2BGR);
    return dst_image;
}

cv::Mat visualizeDepth(const cv::Mat1f& src_image, const cv::Mat1f& sigma)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, 100);
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);
    cv::cvtColor(dst_image, dst_image, cv::COLOR_BGR2HSV);
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int* pt) -> void {
            p[0] = static_cast<unsigned char>(p[2] * 90.0 / 255);  // H in [0,179]
            p[1] = 255;
            p[2] = static_cast<unsigned char>(-500 * sigma(pt[0], pt[1]) + 255);
        });
    cv::cvtColor(dst_image, dst_image, cv::COLOR_HSV2BGR);
    return dst_image;
}

cv::Mat visualizeSigma(const cv::Mat1f& src_image)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, -500, 255);
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);
    return dst_image;
}

cv::Mat visualizeGradient(const cv::Mat1f& src_image)
{
    cv::Mat dst_image = cv::Mat::zeros(src_image.size(), CV_8UC3);
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int* pt) -> void {
            float gray = src_image(pt[0], pt[1]);
            if (math::isInvalid(gray)) {
                p[0] = 255;
                return;
            }
            p[1] = static_cast<unsigned char>(255 * std::max(gray, 0.0f));
            p[2] = static_cast<unsigned char>(-255 * std::min(gray, 0.0f));
        });
    return dst_image;
}

cv::Mat visualizeAge(const cv::Mat1f& src_image)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, 20);
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);
    return dst_image;
}


void showImage(const std::string& window_name,
    const cv::Mat& ref_gray, const cv::Mat& warped_gray, const cv::Mat& cur_gray,
    const cv::Mat& ref_depth, const cv::Mat& ref_sigma, const cv::Mat& ref_grad)
{
    cv::Mat upper_image, under_image;
    cv::Mat show_image;

    cv::hconcat(std::vector<cv::Mat>{
                    ref_gray,
                    warped_gray,
                    cur_gray},
        upper_image);
    cv::hconcat(std::vector<cv::Mat>{
                    ref_depth,
                    ref_sigma,
                    ref_grad},
        under_image);
    cv::vconcat(upper_image, under_image, show_image);
    cv::imshow(window_name, show_image);
}

}  // namespace Draw
