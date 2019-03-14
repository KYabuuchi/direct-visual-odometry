// 相対姿勢の計算
#include "converter.hpp"
#include "io.hpp"
#include "math.hpp"
#include "matplotlibcpp.h"
#include "params.hpp"
#include "transform.hpp"
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>

class Frame
{
public:
    Frame() {}
    cv::Mat depth_image1;
    cv::Mat depth_image2;
    cv::Mat gray_image1;
    cv::Mat gray_image2;
    cv::Mat1f intrinsic_depth;
    cv::Mat1f intrinsic_rgb;
    int cols;
    int rows;
};

std::vector<Frame> createFramePyramid(
    const int level,
    const cv::Mat& intrinsic_depth,
    const cv::Mat& intrinsic_rgb,
    const cv::Mat& gray_image1,
    const cv::Mat& depth_image1,
    const cv::Mat& gray_image2,
    const cv::Mat& depth_image2)
{
    std::vector<Frame> frames(level);

    cv::Mat leveled_gray_image1 = gray_image1;
    cv::Mat leveled_gray_image2 = gray_image2;
    cv::Mat leveled_depth_image1 = depth_image1;
    cv::Mat leveled_depth_image2 = depth_image2;

    auto pow2 = [](int num) -> int {int n=1;for(int i=0;i<num;i++)n*=2; return n; };

    for (int i = level - 1; i >= 0; i--) {
        int scale = pow2(level - 1 - i);
        frames.at(i).intrinsic_depth = intrinsic_depth / scale;
        frames.at(i).intrinsic_depth(2, 2) = 1;
        frames.at(i).intrinsic_rgb = intrinsic_rgb / scale;
        frames.at(i).intrinsic_rgb(2, 2) = 1;

        frames.at(i).depth_image1 = leveled_depth_image1.clone();
        frames.at(i).depth_image2 = leveled_depth_image2.clone();
        frames.at(i).gray_image1 = leveled_gray_image1.clone();
        frames.at(i).gray_image2 = leveled_gray_image2.clone();

        frames.at(i).rows = leveled_depth_image1.rows;
        frames.at(i).cols = leveled_depth_image1.cols;

        leveled_depth_image1 = Converter::cullImage(leveled_depth_image1);
        leveled_depth_image2 = Converter::cullImage(leveled_depth_image2);
        leveled_gray_image1 = Converter::cullImage(leveled_gray_image1);
        leveled_gray_image2 = Converter::cullImage(leveled_gray_image2);
    }

    return frames;
}

// #define DEBUG

int main(int argc, char* argv[])
{
    int num1 = 0, num2 = 5;
    if (argc == 2)
        num1 = std::atoi(argv[1]) + 5;
    std::cout << num1 << " " << num2 << std::endl;
    assert(0 <= num1 and num1 <= 16);


    // prepare depth-image & gray-image
    io::Loader loader("../data/KINECT_1DEG/info.txt");
    cv::Mat depth_image1, depth_image2;
    cv::Mat color_image1, color_image2;
    cv::Mat gray_image1, gray_image2;
#ifndef DEBUG
    loader.readImages(num1, color_image1, depth_image1);
    loader.readImages(num2, color_image2, depth_image2);
    depth_image1 = Converter::depthNormalize(depth_image1);
    depth_image2 = Converter::depthNormalize(depth_image2);
    gray_image1 = Transform::mapDepthtoGray(depth_image1, Converter::colorNormalize(color_image1));
    gray_image2 = Transform::mapDepthtoGray(depth_image2, Converter::colorNormalize(color_image2));
#else
    depth_image1 = cv::imread("depth01.png", cv::IMREAD_UNCHANGED);
    depth_image2 = cv::imread("depth02.png", cv::IMREAD_UNCHANGED);
    color_image1 = cv::imread("rgb01.png", cv::IMREAD_UNCHANGED);
    color_image2 = cv::imread("rgb02.png", cv::IMREAD_UNCHANGED);
    cv::cvtColor(color_image1, gray_image1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(color_image2, gray_image2, cv::COLOR_BGR2GRAY);
    gray_image1.convertTo(gray_image1, CV_32FC1, 1 / 255.0f);
    gray_image2.convertTo(gray_image2, CV_32FC1, 1 / 255.0f);
    depth_image1.convertTo(depth_image1, CV_32FC1, 1 / 5000.0f);
    depth_image2.convertTo(depth_image2, CV_32FC1, 1 / 5000.0f);
#endif

    assert(gray_image1.type() == CV_32FC1);

    // image window
    cv::namedWindow("gray", cv::WINDOW_NORMAL);
    cv::namedWindow("depth", cv::WINDOW_NORMAL);
    cv::namedWindow("grad", cv::WINDOW_NORMAL);

    cv::Mat1f xi(cv::Mat1f::zeros(6, 1));
    std::vector<std::vector<float>> vector_of_residuals;

    // create frame pyramid
    std::vector<Frame> frames = createFramePyramid(
        5,
        Params::KINECTV2_INTRINSIC_DEPTH,
        Params::KINECTV2_INTRINSIC_RGB,
        gray_image1, depth_image1,
        gray_image2, depth_image2);


    for (int level = 0; level < 3; level++) {
        Frame frame = frames.at(level);
        const int ROW = frame.rows;
        const int COL = frame.cols;
        std::cout << "\nLEVEL: " << level << " ROW: " << ROW << " COL: " << COL << std::endl;

        // gradient gray_image
        cv::Mat gradient_image_x = Converter::gradiate(frame.gray_image2, true);
        cv::Mat gradient_image_y = Converter::gradiate(frame.gray_image2, false);

        // vector of residual
        std::vector<float> residuals;

        // show image
        {
            cv::Mat show_image;
            cv::hconcat(frame.gray_image1, frame.gray_image2, show_image);
            show_image.convertTo(show_image, CV_8UC1, 127, 127);
            cv::imshow("gray", show_image);

            cv::hconcat(frame.depth_image1, frame.depth_image2, show_image);
            show_image.convertTo(show_image, CV_8UC1, 100);
            cv::imshow("depth", show_image);

            cv::hconcat(gradient_image_x, gradient_image_y, show_image);
            show_image.convertTo(show_image, CV_32FC1, 1, 1);
            show_image.convertTo(show_image, CV_8UC1, 127, 1);
            cv::imshow("grad", show_image);

            int key = cv::waitKey(0);
            if (key == 'q')
                return 0;
        }

        // iterate direct-method
        for (int iteration = 0; iteration < 10; iteration++) {
            // A xi + B = 0
            cv::Mat1f A(cv::Mat1f::zeros(0, 6));
            cv::Mat1f B(cv::Mat1f::zeros(0, 1));
            cv::Mat moved_image(frame.depth_image1.size(), CV_32FC1, Converter::INVALID);
            float residual = 0;

            for (int col = 0; col < COL; col += 1) {
                for (int row = 0; row < ROW; row += 1) {
                    cv::Point2f x_i0 = cv::Point2f(col, row);
                    float depth1 = frame.depth_image1.at<float>(x_i0);
                    float depth2 = frame.depth_image2.at<float>(x_i0);
                    if (depth2 < 0.001 or depth1 < 0.001)  //   1[mm]
                        continue;

                    // calc residual
                    cv::Point2f x_i = Transform::warp(xi, x_i0, depth2, frame.intrinsic_depth);
                    if ((!math::isRange(x_i.x, 0, COL)) or (!math::isRange(x_i.y, 0, ROW)))
                        continue;

                    float I_2 = Converter::getColorSubpix(frame.gray_image2, x_i);
                    float I_1 = frame.gray_image1.at<float>(x_i0);
                    moved_image.at<float>(x_i0) = I_2;
                    if (Converter::isInvalid(I_1) or Converter::isInvalid(I_2)) {
                        // std::cout << I_1 << " " << I_2 << std::endl;
                        continue;
                    }

                    // calc jacobian
                    float gx = Converter::getColorSubpix(gradient_image_x, x_i);
                    float gy = Converter::getColorSubpix(gradient_image_y, x_i);
                    if (Converter::isInvalid(gx) or Converter::isInvalid(gy)) {
                        // std::cout << gx << " " << gy << std::endl;
                        continue;
                    }

                    float r = I_2 - I_1;
                    residual += r * r;

                    cv::Mat1f jacobi1(cv::Mat1f::zeros(1, 2));
                    jacobi1 = (cv::Mat1f(1, 2) << gx, gy);

                    // if (math::isEpsilon(jacobi1(0)) or math::isEpsilon(jacobi1(1)))
                    //     continue;

                    cv::Mat1f jacobi2 = cv::Mat1f(cv::Mat1f::zeros(2, 6));
                    {
                        cv::Point3f x_c = cv::Point3f(Transform::backProject(frame.intrinsic_depth, cv::Mat1f(x_i0), depth2));
                        // std::cout << x_i0 << " " << x_c << std::endl;

                        float x = x_c.x, y = x_c.y, z = x_c.z;
                        float fx = frame.intrinsic_depth(0, 0);
                        float fy = frame.intrinsic_depth(1, 1);

                        jacobi2(0, 0) = fx / z;
                        jacobi2(1, 0) = 0;
                        jacobi2(0, 1) = 0;
                        jacobi2(1, 1) = fy / z;
                        jacobi2(0, 2) = -fx * x / z / z;
                        jacobi2(1, 2) = -fy * y / z / z;
                        jacobi2(0, 3) = -fx * x * y / z / z;
                        jacobi2(1, 3) = -fy * (1 + y * y / z / z);
                        jacobi2(0, 4) = fx * (1 + x * x / z / z);
                        jacobi2(1, 4) = fy * x * y / z / z;
                        jacobi2(0, 5) = -fx * y / z;
                        jacobi2(1, 5) = fy * x / z;
                    }

                    // stack coefficient
                    cv::vconcat(A, jacobi1 * jacobi2, A);
                    cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r), B);
                }
            }

            moved_image.convertTo(moved_image, CV_8UC1, 127, 127);
            cv::namedWindow("show", cv::WINDOW_NORMAL);
            cv::imshow("show", moved_image);
            int key = cv::waitKey(0);
            if (key == 'q')
                break;

            // 最小二乗法を解く
            cv::Mat1f xi_update;
            cv::solve(A, -B, xi_update, cv::DECOMP_SVD);
            xi = math::se3::concatenate(xi, xi_update);
            std::cout << "iteration: " << iteration << " r  : " << residual << " update " << cv::norm(xi_update) << std::endl;
            residuals.push_back(residual);
            assert(math::testXi(xi));

        }  // iteration

        std::cout << "\n "
                  << math::se3::exp(xi) << std::endl;
        vector_of_residuals.push_back(residuals);

    }  // level

    namespace plt = matplotlibcpp;
    for (size_t i = 0; i < vector_of_residuals.size(); i++) {
        plt::subplot(1, vector_of_residuals.size(), i + 1);
        plt::plot(vector_of_residuals.at(i));
    }
    plt::show(false);
    int key = -1;
    while (key != 'q')
        key = cv::waitKey(10);
}