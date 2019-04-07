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

std::pair<cv::Mat1f, cv::Mat1f> mapDepthtoGray(const cv::Mat1f& depth_image, const cv::Mat1f& gray_image)
{
    cv::Mat1f mapped_image(depth_image.size(), math::INVALID);   // math::INVALID
    cv::Mat1f sigma_image(cv::Mat1f::ones(depth_image.size()));  // 1[m]

    mapped_image.forEach(
        [&](float& p, const int pt[2]) -> void {
            float depth = depth_image(pt[0], pt[1]);
            if (math::isEpsilon(depth))
                return;

            cv::Mat1f x_c = backProject(Params::DEPTH().intrinsic, Convert::toMat1f(pt[1], pt[0]), depth);
            x_c = transform(Params::EXT().invT(), x_c);
            cv::Mat1f x_i = project(Params::RGB().intrinsic, x_c);

            float gray = Convert::getColorSubpix(gray_image, cv::Point2f(x_i));
            sigma_image(pt[0], pt[1]) = 0.1f;

            p = gray;
        });

    return {mapped_image, sigma_image};
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
cv::Mat warpImage(const cv::Mat1f& xi, const cv::Mat1f& gray_image, const cv::Mat1f& depth_image, const cv::Mat1f& intrinsic_matrix)
{
    cv::Mat1f warped_image(gray_image.size(), math::INVALID);
    const int COL = depth_image.cols;
    const int ROW = depth_image.rows;

    for (int x = 0; x < COL; x++) {
        for (int y = 0; y < ROW; y++) {
            cv::Point2i x_i(x, y);
            float depth = depth_image(x_i);
            if (math::isEpsilon(depth))
                continue;

            cv::Point2f warped_x_i = warp(xi, x_i, depth, intrinsic_matrix);
            if (not math::inRange(warped_x_i, depth_image.size()))
                continue;

            float gray = gray_image(x_i);
            warped_image(warped_x_i) = gray;
        }
    }
    return warped_image;
}  // namespace Transform


}  // namespace Transform