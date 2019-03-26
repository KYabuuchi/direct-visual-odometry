#include "core/transform.hpp"
#include "core/convert.hpp"
#include "core/math.hpp"
#include "core/params.hpp"

namespace Transform
{
cv::Mat1f transform(const cv::Mat1f& T, const cv::Mat1f& x)
{
    if (T.size() == cv::Size(4, 4))
        return cv::Mat1f(T.colRange(0, 3).rowRange(0, 3) * x + T.col(3).rowRange(0, 3));
    if (T.size() == cv::Size(1, 6))
        return transform(math::se3::exp(T), x);

    std::cout << "invalid transform " << T.size() << std::endl;
    abort();
}

cv::Mat1f project(const cv::Mat1f& intrinsic, const cv::Mat1f& point)
{
    return Convert::toMat1f(point(0) * intrinsic(0, 0) / point(2) + intrinsic(0, 2), point(1) * intrinsic(1, 1) / point(2) + intrinsic(1, 2));
}

cv::Mat1f backProject(const cv::Mat1f& intrinsic, const cv::Mat1f& point, float depth)
{
    return Convert::toMat1f(depth * (point(0) - intrinsic(0, 2)) / intrinsic(0, 0), depth * (point(1) - intrinsic(1, 2)) / intrinsic(1, 1), depth);
}

cv::Mat mapDepthtoGray(const cv::Mat& depth_image, const cv::Mat& gray_image)
{
    assert(depth_image.type() == CV_32FC1);
    assert(gray_image.type() == CV_32FC1);

    cv::Mat mapped_image = cv::Mat::zeros(depth_image.size(), CV_32FC1);

    mapped_image.forEach<float>(
        [=](float& p, const int position[2]) -> void {
            float depth = depth_image.at<float>(position[0], position[1]);
            if (depth < 1e-6f) {
                p = Convert::INVALID;
                return;
            }
            cv::Mat1f x_c = backProject(Params::depth_intrinsic.intrinsic, Convert::toMat1f(static_cast<float>(position[1]), static_cast<float>(position[0])), depth);
            x_c = transform(Params::extrinsic.invT(), x_c);
            cv::Mat1f x_i = project(Params::rgb_intrinsic.intrinsic, x_c);
            float gray = Convert::getColorSubpix(gray_image, cv::Point2f(x_i));
            if (gray <= 0)
                p = Convert::INVALID;
            else
                p = gray;
        });

    return mapped_image;
}

// warp先の座標を返す
cv::Point2f warp(const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth, const cv::Mat1f& intrinsic_matrix)
{
    cv::Mat1f x_c = backProject(intrinsic_matrix, Convert::toMat1f(x_i.x, x_i.y), depth);
    cv::Mat1f transformed_x_c = transform(xi, x_c);
    cv::Mat1f transformed_x_i = project(intrinsic_matrix, transformed_x_c);
    // std::cout << transformed_x_c.t() << " " << x_c.t() << std::endl;
    return cv::Point2f(transformed_x_i);
}

// warpした画像を返す
cv::Mat warpImage(const cv::Mat1f& xi, const cv::Mat& gray_image, const cv::Mat& depth_image, const cv::Mat1f& intrinsic_matrix)
{
    cv::Mat warped_gray_image = cv::Mat(gray_image.size(), gray_image.type(), Convert::INVALID);
    const int COL = depth_image.cols;
    const int ROW = depth_image.rows;

    for (int x = 0; x < COL; x++) {
        for (int y = 0; y < ROW; y++) {
            cv::Point2i x_i(x, y);
            float depth = depth_image.at<float>(x_i);
            if (depth < math::EPSILON)
                continue;
            cv::Point2i warped_x_i = warp(xi, x_i, depth, intrinsic_matrix);
            if (warped_x_i.x < 0 or warped_x_i.x >= COL or warped_x_i.y < 0 or warped_x_i.y >= ROW)
                continue;

            warped_gray_image.at<float>(warped_x_i) = gray_image.at<float>(x_i);
            // std::cout << warped_x_i << " " << x_i << std::endl;

            //             if (warped_gray_image.at<float>(warped_x_i) == Convert::INVALID)
            //                 warped_gray_image.at<float>(warped_x_i) = gray_image.at<float>(x_i);
            //             else
            //                 warped_gray_image.at<float>(warped_x_i) = (gray_image.at<float>(x_i) + warped_gray_image.at<float>(warped_x_i)) / 2;
        }
    }
    return warped_gray_image;
}  // namespace Transform


}  // namespace Transform