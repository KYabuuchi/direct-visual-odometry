#include "core/convert.hpp"
#include "core/params.hpp"
#include "math/math.hpp"
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
            p = getSubpixel(src_image, {position[1] * reduction, position[0] * reduction});
            // p = src_image.at<float>(position[0] * reduction, position[1] * reduction);
        });

    return culled_image;
}

cv::Mat1f cullIntrinsic(const cv::Mat1f& intrinsic, int times)
{
    if (times == 0)
        return cv::Mat1f(intrinsic);
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
    cv::Mat1f gradiate_image(size, math::INVALID);

    if (x) {
        gradiate_image.forEach(
            [=](float& p, const int pt[2]) -> void {
                if (pt[1] - 1 <= -1 or pt[1] + 1 >= size.width)
                    return;

                float x0 = getSubpixel(gray_image, {pt[1] - 1, pt[0]});
                float x1 = getSubpixel(gray_image, {pt[1] + 1, pt[0]});
                if (isInvalid(x0) or isInvalid(x1)) {
                    return;
                }
                p = x1 - x0;
            });
    } else {
        gradiate_image.forEach(
            [=](float& p, const int pt[2]) -> void {
                if (pt[0] - 1 <= -1 or pt[0] + 1 >= size.height)
                    return;

                float y0 = getSubpixel(gray_image, {pt[1], pt[0] - 1});
                float y1 = getSubpixel(gray_image, {pt[1], pt[0] + 1});
                if (isInvalid(y0) or isInvalid(y1)) {
                    return;
                }
                p = y1 - y0;
            });
    }
    return gradiate_image;
}

float getSubpixel(const cv::Mat1f& img, cv::Point2f pt)
{
    using namespace math;
    auto inRange = generateInRange(img.size());

    int x0 = (int)pt.x;
    int y0 = (int)pt.y;

    if (not inRange({x0, y0}))
        return INVALID;

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float h = pt.x - (float)x0;
    float v = pt.y - (float)y0;

    std::array<float, 4> g = {INVALID, INVALID, INVALID, INVALID};
    g[0] = g[1] = g[2] = g[3] = img(y0, x0);

    if (inRange({x1, y0}))
        g[1] = img(y0, x1);
    if (inRange({x0, y1}))
        g[2] = img(y1, x0);
    if (inRange({x1, y1}))
        g[3] = img(y1, x1);

    int valid = 0;
    int id = 0;
    float last = -1;
    while (true) {
        if (isValid(g[id])) {
            valid++;
            last = g[id];
        } else if (last > 0) {
            g[id] = last;
            valid++;
        }

        if (valid == 4)
            break;
        if (id == 3 and valid == 0) {  // 全部invalid
            return INVALID;
        }
        id = (id + 1) % 4;
    }

    return (g[0] * (1.f - h) + g[1] * h) * (1.f - v)
           + (g[2] * (1.f - h) + g[3] * h) * v;
}

}  // namespace Convert