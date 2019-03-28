#include "core/draw.hpp"
#include "core/convert.hpp"

namespace Draw
{
// 無効画素は赤色へ
cv::Mat visiblizeGrayImage(const cv::Mat& src_image)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, 255);            // change value-depth
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);  // change number of channel
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int* position) -> void {
            if (src_image.at<float>(position[0], position[1]) > -2)
                return;
            p[2] = 255;
        });
    return dst_image;
}

// 無効画素は赤色へ
cv::Mat visiblizeDepthImage(const cv::Mat& src_image)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, 100);            // change value-depth
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);  // change number of channel
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int* position) -> void {
            if (src_image.at<float>(position[0], position[1]) > -2)
                return;
            p[2] = 255;
        });
    return dst_image;
}

// 無効画素は赤色へ
cv::Mat visiblizeGradientImage(const cv::Mat& x_image, const cv::Mat& y_image)
{
    cv::Mat dst_image = cv::Mat::zeros(x_image.size(), CV_8UC3);
    cv::Mat normalized_x_image, normalized_y_image;
    x_image.convertTo(normalized_x_image, CV_8UC1, 127, 127);
    y_image.convertTo(normalized_y_image, CV_8UC1, 127, 127);
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int* position) -> void {
            p[0] = p[1] = 0;
            p[2] = 255;
            if (x_image.at<float>(position[0], position[1]) < -1) {
                return;
            }
            if (y_image.at<float>(position[0], position[1]) < -1) {
                return;
            }
            p[0] = normalized_x_image.at<unsigned char>(position[0], position[1]);
            p[1] = normalized_y_image.at<unsigned char>(position[0], position[1]);
            p[2] = 0;
        });
    return dst_image;
}

// window名,画像x5
void showImage(const std::string& window_name, const cv::Mat1f& pre_gray, const cv::Mat1f& pre_depth,
    const cv::Mat1f& warped_gray, const cv::Mat1f& cur_gray, const cv::Mat1f& cur_depth)
{
    cv::Mat upper_image, under_image;
    cv::Mat show_image;

    cv::hconcat(std::vector<cv::Mat>{
                    Draw::visiblizeGrayImage(pre_gray),
                    Draw::visiblizeGrayImage(warped_gray),
                    Draw::visiblizeGrayImage(cur_gray)},
        upper_image);
    cv::hconcat(std::vector<cv::Mat>{
                    Draw::visiblizeDepthImage(pre_depth),
                    cv::Mat::zeros(pre_depth.size(), CV_8UC3),
                    Draw::visiblizeDepthImage(cur_depth),
                },
        under_image);
    cv::vconcat(upper_image, under_image, show_image);
    cv::imshow(window_name, show_image);
}

}  // namespace Draw
