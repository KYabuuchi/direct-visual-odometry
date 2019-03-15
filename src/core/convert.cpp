#include "core/convert.hpp"
#include "core/math.hpp"
#include "core/params.hpp"
#include <cassert>

namespace Convert
{

cv::Mat1f toMat1f(float x, float y)
{
    return cv::Mat1f(2, 1) << x, y;
}

cv::Mat1f toMat1f(float x, float y, float z)
{
    return cv::Mat1f(3, 1) << x, y, z;
}

cv::Mat depthNormalize(const cv::Mat& depth_image)
{
    cv::Mat tmp_image;
    depth_image.convertTo(tmp_image, CV_32FC1, 1.0 / 5000.0);  // [mm]
    return tmp_image;
}

cv::Mat colorNormalize(const cv::Mat& color_image)
{
    cv::Mat tmp_image;
    cv::cvtColor(color_image, tmp_image, cv::COLOR_BGR2GRAY);
    tmp_image.convertTo(tmp_image, CV_32FC1, 1.0 / 255.0);  // 0~1

    return tmp_image;
}

// T(4x4) => T(4x4)
cv::Mat1f inversePose(const cv::Mat1f& T)
{
    cv::Mat1f tmp(cv::Mat1f::zeros(4, 4));
    cv::Mat1f inverse_R(T.colRange(0, 3).rowRange(0, 3).t());
    cv::Mat1f inverse_t(-T.col(3).rowRange(0, 3));
    inverse_R.copyTo(tmp.colRange(0, 3).rowRange(0, 3));
    inverse_t.copyTo(tmp.col(3).rowRange(0, 3));
    return tmp;
}

// 勾配を計算(欠けていたら0)
cv::Mat1f gradiate(const cv::Mat1f& gray_image, bool x)
{
    assert(gray_image.type() == CV_32FC1);

    cv::Size size = gray_image.size();
    cv::Mat gradiate_image = cv::Mat(gray_image.size(), CV_32FC1, INVALID);

    if (x) {
        gradiate_image.forEach<float>(
            [=](float& p, const int position[2]) -> void {
                if (position[1] <= 1 or position[1] + 1 >= size.width)
                    return;

                float x0 = gray_image.at<float>(position[0], position[1]);
                float x1 = gray_image.at<float>(position[0], position[1] + 1);
                if (x0 < 0 or x1 < 0)
                    return;
                p = x1 - x0;
            });
    } else {
        gradiate_image.forEach<float>(
            [=](float& p, const int position[2]) -> void {
                if (position[0] <= 1 or position[0] + 1 >= size.height)
                    return;

                float y0 = gray_image.at<float>(position[0], position[1]);
                float y1 = gray_image.at<float>(position[0] + 1, position[1]);
                if (y0 < 0 or y1 < 0)
                    return;

                p = y1 - y0;
            });
    }
    return gradiate_image;
}

// 画素を計算(欠けていたらINVALID)
float getColorSubpix(const cv::Mat1f& img, cv::Point2f pt)
{
    assert(img.type() == CV_32FC1);

    if (pt.x < 0 || pt.x > img.size().width, pt.y < 0 || pt.y > img.size().height)
        return INVALID;

    int x = (int)pt.x;
    int y = (int)pt.y;

    int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);

    float a = pt.x - (float)x;
    float c = pt.y - (float)y;

    float v00 = img(y0, x0);
    float v01 = img(y0, x1);
    float v10 = img(y1, x0);
    float v11 = img(y1, x1);

    if (isValid(v00) and isValid(v01) and isValid(v10) and isValid(v11))
        return (v00 * (1.f - a) + v01 * a) * (1.f - c)
               + (v10 * (1.f - a) + v11 * a) * c;

    return INVALID;
}

cv::Mat cullImage(const cv::Mat& src_image)
{
    assert(src_image.type() == CV_32FC1);
    cv::Mat1f culled_image = cv::Mat::zeros(src_image.size() / 2, CV_32FC1);
    culled_image.forEach(
        [=](float& p, const int position[2]) -> void {
            p = src_image.at<float>(position[0] * 2, position[1] * 2);
        });

    return culled_image;
}
}  // namespace Convert