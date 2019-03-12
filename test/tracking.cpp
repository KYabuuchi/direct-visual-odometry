// 相対姿勢の計算(without using image-pyramid)
#include "converter.hpp"
#include "io.hpp"
#include "math.hpp"
#include "params.hpp"
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>

//#define DEBUG3
//#define DEBUGG

// warp先の画素値を返す
cv::Point2f warp(const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth)
{
    cv::Mat1f x_c = Converter::backProject(Params::KINECTV2_INTRINSIC_DEPTH, Converter::toMat1f(x_i.x, x_i.y), depth);
    x_c = Converter::transformByXi(xi, x_c);
    cv::Mat1f transformed_x_i = Converter::project(Params::KINECTV2_INTRINSIC_DEPTH, x_c);
    return cv::Point2f(transformed_x_i);
}

// 勾配を計算(欠けていたら0)
cv::Mat gradiate(const cv::Mat& gray_image, bool x)
{
    cv::Size size = gray_image.size();
    cv::Mat gradiate_image = cv::Mat::zeros(gray_image.size(), CV_32FC1);

    if (x) {
        gradiate_image.forEach<float>(
            [=](float& p, const int position[2]) -> void {
                if (position[0] == 0 or position[0] == size.width)
                    return;

                float x0 = gray_image.at<float>(position[0] - 1, position[1]);
                float x1 = gray_image.at<float>(position[0] + 1, position[1]);
                if (x0 == 0 or x1 == 0)
                    return;
                p = x1 - x0;
            });
    } else {
        gradiate_image.forEach<float>(
            [=](float& p, const int position[2]) -> void {
                if (position[1] == 0 or position[1] == size.height)
                    return;

                float y0 = gray_image.at<float>(position[0], position[1] - 1);
                float y1 = gray_image.at<float>(position[0], position[1] + 1);
                if (y0 == 0 or y1 == 0)
                    return;
                p = y1 - y0;
            });
    }
    return gradiate_image;
}

struct Camera {
    float fx;
    float fy;
    float cx;
    float cy;
};

int main(int argc, char* argv[])
{
    int num1 = 3;
    int num2 = 5;
    if (argc == 2) {
        int tmp = std::atoi(argv[1]);
        num1 = tmp + 5;
        // num2 = tmp > 0 ? tmp : 0;
    }

    std::cout << num1 << " " << num2 << std::endl;

    // prepare depth-image & gray-image
#ifndef DEBUGG
    io::Loader loader("../data/KINECT_5MM/info.txt");
    cv::Mat depth_image1, color_image1;
    cv::Mat depth_image2, color_image2;
    loader.readImages(num1, color_image1, depth_image1);
    loader.readImages(num2, color_image2, depth_image2);
    depth_image1 = Converter::depthNormalize(depth_image1);
    depth_image2 = Converter::depthNormalize(depth_image2);
    cv::Mat gray_image1 = Converter::mapDepthtoGray(depth_image1, Converter::colorNormalize(color_image1));
    cv::Mat gray_image2 = Converter::mapDepthtoGray(depth_image2, Converter::colorNormalize(color_image2));
#else
    cv::Mat depth_image1 = cv::imread("depth.png", cv::IMREAD_UNCHANGED);
    cv::Mat depth_image2 = cv::imread("depth.png", cv::IMREAD_UNCHANGED);
    cv::Mat gray_image1 = cv::imread("gray1.png", cv::IMREAD_UNCHANGED);
    cv::Mat gray_image2 = cv::imread("gray2.png", cv::IMREAD_UNCHANGED);
    gray_image1.convertTo(gray_image1, CV_32FC1, 1.0 / 255.0);  // 0~1
    gray_image2.convertTo(gray_image2, CV_32FC1, 1.0 / 255.0);  // 0~1
    depth_image1 = Converter::depthNormalize(depth_image1);
    depth_image2 = Converter::depthNormalize(depth_image2);
#endif

    assert(gray_image1.type() == CV_32FC1);

    // iterate direct-method
    const int COL = gray_image1.cols;
    const int ROW = gray_image1.rows;

    // camera model
    Camera camera{
        Params::KINECTV2_INTRINSIC_DEPTH(0, 0),
        Params::KINECTV2_INTRINSIC_DEPTH(1, 1),
        Params::KINECTV2_INTRINSIC_DEPTH(0, 2),
        Params::KINECTV2_INTRINSIC_DEPTH(1, 2)};

    // gradient gray_image
    cv::Mat gradient_image_x = gradiate(gray_image2, true);
    cv::Mat gradient_image_y = gradiate(gray_image2, false);

// A xi + B = 0
#ifdef DEBUG3
    cv::Mat1f A(cv::Mat1f::zeros(0, 3));
#else
    cv::Mat1f A(cv::Mat1f::zeros(0, 6));
#endif

    cv::Mat1f B(cv::Mat1f::zeros(0, 1));
    cv::Mat1f xi = math::se3::log(cv::Mat1f::eye(4, 4));

    for (int level = 0; level < 5; level++) {
        int skip = 20 * (5 - level);
        std::cout << "SKIP " << skip << std::endl;
        for (int iteration = 0; iteration < 10; iteration++) {
            float residual = 0;
            for (int col = 0; col < COL; col += skip) {
                for (int row = 0; row < ROW; row += skip) {
                    cv::Point2f x_i0 = cv::Point2f(col, row);
                    float depth = depth_image2.at<float>(x_i0);
                    if (depth < 0.01)  // 1e-6[m]
                        continue;

                    // calc residual
                    cv::Point2f x_i = warp(xi, x_i0, depth);
                    float a = Converter::getColorSubpix(gray_image2, x_i);
                    float b = gray_image1.at<float>(x_i0);
                    float r = a - b;
                    residual += r * r;

                    if (r == 0)
                        continue;

                    // clang-format off
                cv::Mat1f jacobian1 = (cv::Mat1f(1, 2) << 
                    gradient_image_x.at<float>(x_i),
                    gradient_image_y.at<float>(x_i));
                    // clang-format on

                    cv::Point3f x_c = cv::Point3f(Converter::backProject(Params::KINECTV2_INTRINSIC_DEPTH, cv::Mat1f(x_i0), depth));
#ifdef DEBUG3
                    cv::Mat1f jacobian2 = cv::Mat1f::zeros(2, 3);
#else
                    cv::Mat1f jacobian2 = cv::Mat1f::zeros(2, 6);
#endif
                    {
                        const Camera& c = camera;
                        float x = x_c.x;
                        float y = x_c.y;
                        float z = x_c.z;
                        jacobian2(0, 0) = c.fx / z;
                        jacobian2(0, 1) = 0;
                        jacobian2(0, 2) = -c.fx * x / z / z;
                        jacobian2(1, 0) = 0;
                        jacobian2(1, 1) = c.fy / z;
                        jacobian2(1, 2) = -c.fy * y / z / z;
#ifndef DEBUG3
                        jacobian2(0, 3) = -c.fx * x * y / z / z;
                        jacobian2(0, 4) = c.fx * (1 + x * x / z / z);
                        jacobian2(0, 5) = -c.fx * y / z;
                        jacobian2(1, 3) = -c.fy * (1 + y * y / z / z);
                        jacobian2(1, 4) = c.fy * x * y / z / z;
                        jacobian2(1, 5) = c.fy * x / z;
#endif
                    }

                    // stack coefficient
                    cv::vconcat(A, jacobian1 * jacobian2, A);
                    cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r), B);
                }
            }

            cv::Mat update;
            // A xi = - B
            cv::solve(A, -B, update, cv::DECOMP_SVD);
            // std::cout << A << std::endl;

#ifndef DEBUG3
            xi = math::se3::concatenate(xi, update);
            std::cout << xi.t() << std::endl;
#endif
            std::cout << residual << std::endl;
            std::cout << update.t() << std::endl;
        }
    }
    std::cout << "\n"
              << math::se3::exp(xi) << std::endl;

    // show image
    cv::Mat show_image;
    cv::hconcat(gray_image1, gray_image2, show_image);
    show_image.convertTo(show_image, CV_8UC1, 255);
    cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::imshow("window", show_image);

    cv::hconcat(gradient_image_x, gradient_image_y, show_image);
    show_image.convertTo(show_image, CV_8UC1, 255);
    cv::imshow("grad", show_image);
    cv::waitKey(0);
}