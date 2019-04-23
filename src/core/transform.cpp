#include "core/transform.hpp"
#include "core/convert.hpp"
#include "core/params.hpp"
#include "math/math.hpp"

namespace Transform
{
cv::Point3f transform(const cv::Mat1f& T, const cv::Point3f& x)
{
    if (T.size() == cv::Size(4, 4)) {
        cv::Mat transformed = T.colRange(0, 3).rowRange(0, 3) * cv::Mat(x) + T.col(3).rowRange(0, 3);
        return cv::Point3f(transformed);
    }
    if (T.size() == cv::Size(1, 6))
        return transform(math::se3::exp(T), x);

    std::cout << "invalid transform " << T.size() << std::endl;
    abort();
}

cv::Point2f project(const cv::Mat1f& K, const cv::Point3f& point)
{
    return cv::Point2f(point.x * K(0, 0) / point.z + K(0, 2), point.y * K(1, 1) / point.z + K(1, 2));
}

cv::Point3f backProject(const cv::Mat1f& K, const cv::Point2f& point, float depth)
{
    return cv::Point3f(depth * (point.x - K(0, 2)) / K(0, 0), depth * (point.y - K(1, 2)) / K(1, 1), depth);
}

// warp先の座標を返す
cv::Point2f warp(const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth, const cv::Mat1f& K)
{
    cv::Point3f x_c = backProject(K, x_i, depth);
    cv::Point3f transformed_x_c = transform(xi, x_c);
    return project(K, transformed_x_c);
}

// warpした画像を返す
cv::Mat warpImage(const cv::Mat1f& xi, const cv::Mat1f& gray_image, const cv::Mat1f& depth_image, const cv::Mat1f& K)
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

            cv::Point2f warped_x_i = warp(xi, x_i, depth, K);
            if (not math::inRange(warped_x_i, depth_image.size()))
                continue;

            float gray = Convert::getSubpixel(gray_image, x_i);
            warped_image(warped_x_i) = gray;
        }
    }
    return warped_image;
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

            cv::Point3f x_c = backProject(Params::DEPTH().intrinsic, {pt[1], pt[0]}, depth);
            x_c = transform(Params::EXT().invT(), x_c);
            cv::Point2f x_i = project(Params::RGB().intrinsic, x_c);

            float gray = Convert::getSubpixel(gray_image, x_i);
            sigma_image(pt[0], pt[1]) = 0.1f;

            p = gray;
        });

    return {mapped_image, sigma_image};
}



}  // namespace Transform