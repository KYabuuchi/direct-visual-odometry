#include "core/draw.hpp"
#include "core/convert.hpp"
#include "math/math.hpp"

namespace Draw
{
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

cv::Mat visualizeDepthRaw(const cv::Mat1f& src_image)
{
    cv::Mat dst_image = src_image.clone();
    dst_image = cv::min(dst_image, 2.5f);
    dst_image = cv::max(dst_image, 0.5f);
    dst_image.convertTo(dst_image, CV_8UC1, 100);
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);
    return dst_image;
}

cv::Mat visualizeDepth(const cv::Mat1f& src_image)
{
    cv::Mat dst_image = cv::Mat(src_image.size(), CV_8UC3);
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int pt[2]) -> void {
            float tmp = src_image(pt[0], pt[1]);
            if (tmp < math::EPSILON) {
                p[0] = p[1] = p[2] = 0;
                return;
            }
            tmp = std::clamp((tmp - 0.70f) * 70.0f, 0.0f, 180.0f);
            p[0] = static_cast<unsigned char>(tmp);  // H in [0,179]
            p[1] = 255;
            p[2] = 255;
        });
    cv::cvtColor(dst_image, dst_image, cv::COLOR_HSV2BGR);
    return dst_image;
}

cv::Mat visualizeDepth(const cv::Mat1f& src_image, const cv::Mat1f& sigma)
{
    cv::Mat dst_image = cv::Mat(src_image.size(), CV_8UC3);
    dst_image.forEach<cv::Vec3b>(
        [&](cv::Vec3b& p, const int* pt) -> void {
            float tmp = src_image(pt[0], pt[1]);
            if (tmp < math::EPSILON) {
                p[0] = p[1] = p[2] = 0;
                return;
            }
            tmp = std::clamp((tmp - 0.70f) * 70.0f, 0.0f, 180.0f);
            p[0] = static_cast<unsigned char>(tmp);  // H in [0,179]
            p[1] = 255;
            p[2] = static_cast<unsigned char>(-500 * std::min(sigma(pt[0], pt[1]), 0.5f) + 255);
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
    src_image.convertTo(dst_image, CV_8UC1, 10);
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);
    return dst_image;
}

cv::Mat merge(std::vector<cv::Mat>& tail)
{
    size_t N = tail.size();
    if (N == 1) {
        const cv::Mat& img = tail[0];
        cv::Mat zero = cv::Mat::zeros(img.size(), img.type());
        cv::Mat m;
        cv::vconcat(img, zero, m);
        return m;
    }
    if (N == 2) {
        cv::Mat m;
        cv::vconcat(tail[0], tail[1], m);
        return m;
    }
    {
        cv::Mat m1;
        cv::vconcat(tail[N - 2], tail[N - 1], m1);
        tail.pop_back();
        tail.pop_back();
        cv::Mat m2 = merge(tail);
        cv::Mat m3;
        cv::hconcat(m1, m2, m3);
        return m3;
    }
}

void showImage(const std::string& window_name, std::vector<cv::Mat>& tail)
{
    cv::Mat show_image = merge(tail);
    cv::imshow(window_name, show_image);
}


}  // namespace Draw
