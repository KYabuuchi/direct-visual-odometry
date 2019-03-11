// 相対姿勢の計算(without using image-pyramid)
#include "converter.hpp"
#include "io.hpp"
#include "math.hpp"
#include "matplotlibcpp.h"
#include "params.hpp"
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace plt = matplotlibcpp;

// warp先の画素値を返す
cv::Point2f warp(const cv::Mat& gray_image, const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth)
{
    cv::Mat1f x_c = Converter::backProject(Params::KINECTV2_INTRINSIC_DEPTH, Converter::toMat1f(x_i.x, x_i.y), depth);
    x_c = Converter::transformByXi(xi, x_c);
    cv::Mat1f transformed_x_i = Converter::project(Params::KINECTV2_INTRINSIC_DEPTH, x_c);
    return cv::Point2f(transformed_x_i);
}

struct Camera {
    float fx;
    float fy;
    float cx;
    float cy;
};

int main(int argc, char* argv[])
{
    int num = 1;
    if (argc == 2)
        num = std::atoi(argv[1]);

    // prepare depth-image & gray-image
    io::Loader loader("../data/KINECT_1DEG/info.txt");
    cv::Mat depth_image1, color_image1;
    cv::Mat depth_image2, color_image2;
    loader.readImages(0, color_image1, depth_image1);
    loader.readImages(num, color_image2, depth_image2);
    depth_image1 = Converter::depthNormalize(depth_image1);
    depth_image2 = Converter::depthNormalize(depth_image2);
    cv::Mat gray_image1 = Converter::mapDepthtoGray(depth_image1, Converter::colorNormalize(color_image1));
    cv::Mat gray_image2 = Converter::mapDepthtoGray(depth_image2, Converter::colorNormalize(color_image2));

    // iterate direct-method
    const int COL = gray_image1.cols;
    const int ROW = gray_image1.rows;

    // camera model
    Camera camera{
        Params::KINECTV2_INTRINSIC_DEPTH(0, 0),
        Params::KINECTV2_INTRINSIC_DEPTH(1, 1),
        Params::KINECTV2_INTRINSIC_DEPTH(0, 2),
        Params::KINECTV2_INTRINSIC_DEPTH(1, 2)};

    std::vector<float> residuals;

    cv::Mat1f xi(cv::Mat1f::zeros(6, 1));
    // A xi + B = 0
    cv::Mat1f A(cv::Mat1f::zeros(0, 6));
    cv::Mat1f B(cv::Mat1f::zeros(0, 1));

    // gradient gray_image
    cv::Mat gradient_image_x;
    cv::Mat gradient_image_y;
    cv::Sobel(gray_image2, gradient_image_x, CV_32FC1, 1, 0, 3);
    cv::Sobel(gray_image2, gradient_image_y, CV_32FC1, 0, 1, 3);

    for (int iteration = 0; iteration < 1; iteration++) {
        float residual = 0;
        for (int x = 0; x < COL / 10; x++) {
            for (int y = 0; y < ROW / 10; y++) {
                float depth = depth_image2.at<float>(y, x);
                if (depth < 1e-6)  // 1e-6[mm]
                    continue;

                // calc residual
                cv::Point2f x_i0 = cv::Point2f(x, y);
                cv::Point2f x_i = warp(gray_image2, xi, x_i0, depth);
                float a = Converter::getColorSubpix(gray_image2, x_i);
                float b = gray_image1.at<float>(x_i0);
                float r = a - b;
                residual += r * r;

                // calc jacobian
                // clang-format off
                cv::Mat1f jacobian1 = (cv::Mat1f(1, 2) << 
                    gradient_image_x.at<float>(x_i),
                    gradient_image_y.at<float>(x_i));
                // clang-format on

                cv::Point3f x_c = cv::Point3f(Converter::backProject(Params::KINECTV2_INTRINSIC_DEPTH, cv::Mat1f(x_i0), depth));
                cv::Mat1f jacobian2 = cv::Mat1f::zeros(2, 6);
                {
                    Camera& c = camera;
                    float x = x_c.x, y = x_c.y, z = x_c.z;
                    jacobian2(0, 0) = c.fx / z;
                    jacobian2(0, 1) = 0;
                    jacobian2(0, 2) = c.fx * x / z / z;
                    jacobian2(0, 3) = -c.fx * x * y / z / z;
                    jacobian2(0, 4) = c.fx * (1 + x * x / z / z);
                    jacobian2(0, 5) = -c.fx * y / z;
                    jacobian2(1, 0) = 0;
                    jacobian2(1, 1) = c.fy / z;
                    jacobian2(1, 2) = c.fy * y / z / z;
                    jacobian2(1, 3) = -c.fy * (1 + y * y / z / z);
                    jacobian2(1, 4) = c.fy * x * y / z / z;
                    jacobian2(1, 5) = c.fy * x / z;
                }

                // stack coefficient
                cv::vconcat(A, jacobian1 * jacobian2, A);
                cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << b), B);
            }
        }
        std::cout << residual << std::endl;
        residuals.push_back(residual);
    }
    cv::Mat update;
    cv::solve(A, B, update, cv::DECOMP_SVD);
    std::cout << update << std::endl;

    // show graph
    // plt::plot(residuals);
    // plt::show(false);

    // show image
    cv::Mat show_image;
    cv::hconcat(gray_image1, gray_image2, show_image);
    show_image.convertTo(show_image, CV_8UC1, 255);
    cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::imshow("window", show_image);
    cv::waitKey(0);

    // cv::hconcat(gradient_image_x, gradient_image_y, show_image);
    // show_image.convertTo(show_image, CV_8UC1, 255);
    // cv::imshow("window", show_image);
    // cv::waitKey(0);
}