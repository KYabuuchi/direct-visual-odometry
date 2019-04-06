#include "core/convert.hpp"
#include "core/math.hpp"
#include "core/params.hpp"
#include <cassert>

namespace Convert
{
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

cv::Mat1f cullImage(const cv::Mat1f& src_image, int times)
{
    if (times == 0)
        return cv::Mat1f(src_image);

    int reduction = math::pow(2, times);
    cv::Mat1f culled_image(cv::Mat::zeros(src_image.size() / reduction, CV_32FC1));
    culled_image.forEach(
        [=](float& p, const int position[2]) -> void {
            p = src_image.at<float>(position[0] * reduction, position[1] * reduction);
        });

    return culled_image;
}

cv::Mat1f cullIntrinsic(const cv::Mat1f& intrinsic, int times)
{
    cv::Mat1f K = cv::Mat1f(intrinsic / math::pow(2, times));
    K(2, 2) = 1;
    return K;
}

cv::Mat1f inversePose(const cv::Mat1f& T)
{
    cv::Mat1f tmp(cv::Mat1f::zeros(4, 4));
    cv::Mat1f inverse_R(T.colRange(0, 3).rowRange(0, 3).t());
    cv::Mat1f inverse_t(-T.col(3).rowRange(0, 3));
    inverse_R.copyTo(tmp.colRange(0, 3).rowRange(0, 3));
    inverse_t.copyTo(tmp.col(3).rowRange(0, 3));
    return tmp;
}

cv::Mat1f gradiate(const cv::Mat1f& gray_image, bool x)
{
    using namespace math;
    cv::Size size = gray_image.size();
    cv::Mat1f gradiate_image(cv::Mat1f::zeros(size));

    if (x) {
        gradiate_image.forEach(
            [=](float& p, const int pt[2]) -> void {
                if (pt[1] - 1 <= -1 or pt[1] + 1 >= size.width)
                    return;

                float x0 = gray_image(pt[0], pt[1] - 1);
                float x1 = gray_image(pt[0], pt[1] + 1);

                if (isInvalid(x0) or isInvalid(x1))
                    return;
                p = x1 - x0;
            });
    } else {
        gradiate_image.forEach(
            [=](float& p, const int pt[2]) -> void {
                if (pt[0] - 1 <= -1 or pt[0] + 1 >= size.height)
                    return;

                float y0 = gray_image(pt[0] - 1, pt[1]);
                float y1 = gray_image(pt[0] + 1, pt[1]);
                if (isInvalid(y0) or isInvalid(y1))
                    return;
                p = y1 - y0;
            });
    }
    return gradiate_image;
}

float getColorSubpix(const cv::Mat1f& img, cv::Point2f pt)
{
    using namespace math;

    int x = (int)pt.x;
    int y = (int)pt.y;
    if (x < 0 || x + 1 >= img.size().width || y < 0 || y + 1 >= img.size().height)
        return math::INVALID;

    int x0 = x;
    int x1 = x + 1;
    int y0 = y;
    int y1 = y + 1;

    float a = pt.x - (float)x;
    float c = pt.y - (float)y;

    float v00 = img(y0, x0);
    float v01 = img(y0, x1);
    float v10 = img(y1, x0);
    float v11 = img(y1, x1);

    if (isValid(v00) and isValid(v01) and isValid(v10) and isValid(v11))
        return (v00 * (1.f - a) + v01 * a) * (1.f - c)
               + (v10 * (1.f - a) + v11 * a) * c;

    return math::INVALID;
}

}  // namespace Convert