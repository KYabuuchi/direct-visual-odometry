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
cv::Point2f warp(const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth)
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
    int num1 = 1;
    int num2 = 1;
    if (argc == 2) {
        int tmp = std::atoi(argv[1]);
        num1 = tmp > 0 ? 0 : -tmp;
        num2 = tmp > 0 ? tmp : 0;
    }

    // prepare depth-image & gray-image
    //io::Loader loader("../data/KINECT_5MM/info.txt");
    cv::Mat depth_image1, color_image1;
    cv::Mat depth_image2, color_image2;
    //loader.readImages(num1, color_image1, depth_image1);
    //loader.readImages(num2, color_image2, depth_image2);
    cv::Mat gray_image1;  // = Converter::mapDepthtoGray(depth_image1, Converter::colorNormalize(color_image1));
    cv::Mat gray_image2;  // = Converter::mapDepthtoGray(depth_image2, Converter::colorNormalize(color_image2));

    depth_image1 = cv::imread("depth.png", cv::IMREAD_UNCHANGED);
    depth_image2 = cv::imread("depth.png", cv::IMREAD_UNCHANGED);
    gray_image1 = cv::imread("gray1.png", cv::IMREAD_UNCHANGED);
    gray_image2 = cv::imread("gray2.png", cv::IMREAD_UNCHANGED);
    depth_image1 = Converter::depthNormalize(depth_image1);
    depth_image2 = Converter::depthNormalize(depth_image2);
    gray_image1.convertTo(gray_image1, CV_32FC1, 1.0 / 255.0);  // 0~1
    gray_image2.convertTo(gray_image2, CV_32FC1, 1.0 / 255.0);  // 0~1


    cv::dilate(gray_image1, gray_image1, cv::Mat(), cv::Point(-1, -1), 1);
    cv::dilate(gray_image2, gray_image2, cv::Mat(), cv::Point(-1, -1), 1);

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
    cv::Mat gradient_image_x;
    cv::Mat gradient_image_y;
    cv::Sobel(gray_image2, gradient_image_x, CV_32FC1, 1, 0, 3);
    cv::Sobel(gray_image2, gradient_image_y, CV_32FC1, 0, 1, 3);

    // A xi + B = 0
    cv::Mat1f A(cv::Mat1f::zeros(0, 6));
    cv::Mat1f B(cv::Mat1f::zeros(0, 1));
    std::vector<float> residuals;

    cv::Mat1f tmp = cv::Mat1f::eye(4, 4);
    cv::Mat1f xi = math::se3::log(tmp);

    for (int iteration = 0; iteration < 5; iteration++) {

        float residual = 0;
        for (int col = 0; col < COL; col += 1) {
            for (int row = 0; row < ROW; row += 10) {
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

                // calc jacobian
                // clang-format off
                cv::Mat1f jacobian1 = (cv::Mat1f(1, 2) << 
                    gradient_image_x.at<float>(x_i),
                    gradient_image_y.at<float>(x_i));
                // clang-format on
                if (jacobian1(0) == 0 and jacobian1(1) == 0)
                    continue;
                // std::cout << jacobian1 << std::endl;

                cv::Point3f x_c = cv::Point3f(Converter::backProject(Params::KINECTV2_INTRINSIC_DEPTH, cv::Mat1f(x_i0), depth));
                cv::Mat1f jacobian2 = cv::Mat1f::zeros(2, 6);
                {
                    const Camera& c = camera;
                    float x = x_c.x;
                    float y = x_c.y;
                    float z = x_c.z;
                    jacobian2(0, 0) = c.fx / z;
                    jacobian2(0, 1) = 0;
                    jacobian2(0, 2) = -c.fx * x / z / z;
                    jacobian2(0, 3) = -c.fx * x * y / z / z;
                    jacobian2(0, 4) = c.fx * (1 + x * x / z / z);
                    jacobian2(0, 5) = -c.fx * y / z;
                    jacobian2(1, 0) = 0;
                    jacobian2(1, 1) = c.fy / z;
                    jacobian2(1, 2) = -c.fy * y / z / z;
                    jacobian2(1, 3) = -c.fy * (1 + y * y / z / z);
                    jacobian2(1, 4) = c.fy * x * y / z / z;
                    jacobian2(1, 5) = c.fy * x / z;
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
        // std::cout << cv::norm(B) << std::endl;
        // std::cout << cv::norm(A * update + B) << std::endl;
        xi = math::se3::concatenate(xi, update);
        residuals.push_back(residual);
        std::cout << residual << std::endl;
        // std::cout << update.t() << std::endl;
        std::cout << xi.t() << std::endl;
    }
    std::cout << math::se3::exp(xi) << std::endl;

    // show graph
    // plt::plot(residuals);
    // plt::show(false);

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