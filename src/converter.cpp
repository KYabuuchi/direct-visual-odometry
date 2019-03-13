#include "converter.hpp"
#include "math.hpp"
#include "params.hpp"
#include <cassert>

namespace Converter
{


float getColorSubpix(const cv::Mat1f& img, cv::Point2f pt)
{
    assert(img.type() == CV_32FC1);

    if (pt.x < 0 || pt.x > img.size().width, pt.y < 0 || pt.y > img.size().height)
        return -1.0f;

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

    if (v00 < 0 or v01 < 0 or v11 < 0 or v10 < 0)
        return -1.0f;

    return (v00 * (1.f - a) + v01 * a) * (1.f - c)
           + (v10 * (1.f - a) + v11 * a) * c;
}

cv::Mat1f toMat1f(float x, float y)
{
    return cv::Mat1f(2, 1) << x, y;
}

cv::Mat1f toMat1f(float x, float y, float z)
{
    return cv::Mat1f(3, 1) << x, y, z;
}

// T(4x4),x(3x1) => Rx+t(3x1)
cv::Mat1f transformByT(const cv::Mat1f T, const cv::Mat1f& x)
{
    assert(T.size() == cv::Size(4, 4));
    return cv::Mat1f(T.colRange(0, 3).rowRange(0, 3) * x + T.col(3).rowRange(0, 3));
}

// xi,x(3x1) => Rx+t(3x1)
cv::Mat1f transformByXi(const cv::Mat1f xi, const cv::Mat1f& x)
{
    return transformByT(math::se3::exp(xi), x);
}

// x_c => x_i
cv::Mat1f project(const cv::Mat1f& intrinsic, const cv::Mat1f& point)
{
    return toMat1f(point(0) * intrinsic(0, 0) / point(2) + intrinsic(0, 2), point(1) * intrinsic(1, 1) / point(2) + intrinsic(1, 2));
}

// x_i => x_c
cv::Mat1f backProject(const cv::Mat1f& intrinsic, const cv::Mat1f& point, float depth)
{
    return toMat1f(depth * (point(0) - intrinsic(0, 2)) / intrinsic(0, 0), depth * (point(1) - intrinsic(1, 2)) / intrinsic(1, 1), depth);
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

// 無効な画素には-1が入る
cv::Mat mapDepthtoGray(const cv::Mat& depth_image, const cv::Mat& gray_image)
{
    assert(depth_image.type() == CV_32FC1);
    assert(gray_image.type() == CV_32FC1);

    cv::Mat mapped_image = cv::Mat::zeros(depth_image.size(), CV_32FC1);

    mapped_image.forEach<float>(
        [=](float& p, const int position[2]) -> void {
            float depth = depth_image.at<float>(position[0], position[1]);
            if (depth < 1e-6f) {
                p = -1.0f;
                return;
            }
            cv::Mat1f x_c = backProject(Params::KINECTV2_INTRINSIC_DEPTH, toMat1f(static_cast<float>(position[1]), static_cast<float>(position[0])), depth);
            x_c = transformByT(Params::KINECTV2_EXTRINSIC, x_c);
            cv::Mat1f x_i = project(Params::KINECTV2_INTRINSIC_RGB, x_c);
            float gray = getColorSubpix(gray_image, cv::Point2f(x_i));
            p = gray;
        });

    return mapped_image;
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

}  // namespace Converter