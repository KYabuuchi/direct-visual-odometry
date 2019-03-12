// 相対姿勢の計算
#include "converter.hpp"
#include "io.hpp"
#include "math.hpp"
#include "params.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#define DEBUG3

#define EPSILON 1e-6

inline bool isEpsilon(float num)
{
    return std::abs(num) < EPSILON;
}

inline bool isRange(float num, int min, int max)
{
    if (num <= min)
        return false;
    if (max <= num)
        return false;
    return true;
}

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

cv::Point2f warp(const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth, const Frame& frame);
cv::Mat gradiate(const cv::Mat& gray_image, bool x);
std::vector<Frame> createFramePyramid(
    const int level,
    const cv::Mat& intrinsic_depth,
    const cv::Mat& intrinsic_rgb,
    const cv::Mat& gray_image1,
    const cv::Mat& depth_image1,
    const cv::Mat& gray_image2,
    const cv::Mat& depth_image2);

cv::Mat1f solveNE(const cv::Mat1f& A, const cv::Mat1f& B)
{
    std::cout << "cv::solve()\n"
              << std::endl;
    cv::Mat show;
    cv::hconcat(A, B, show);
    // std::cout << show << std::endl;

    Eigen::MatrixXf eigen_A, eigen_B;
    cv::cv2eigen(A, eigen_A);
    cv::cv2eigen(B, eigen_B);
    Eigen::MatrixXf eigen_update = eigen_A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-eigen_B);
    cv::Mat1f update;
    cv::eigen2cv(eigen_update, update);
    return update;
}

int main(int argc, char* argv[])
{
    int num1 = 0, num2 = 5;
    if (argc == 2)
        num1 = std::atoi(argv[1]) + 5;
    std::cout << num1 << " " << num2 << std::endl;
    assert(0 <= num1 and num1 <= 16);

    // prepare depth-image & gray-image
    io::Loader loader("../data/KINECT_5MM/info.txt");
    cv::Mat depth_image1, color_image1;
    cv::Mat depth_image2, color_image2;
    loader.readImages(num1, color_image1, depth_image1);
    loader.readImages(num2, color_image2, depth_image2);
    depth_image1 = Converter::depthNormalize(depth_image1);
    depth_image2 = Converter::depthNormalize(depth_image2);
    cv::Mat gray_image1 = Converter::mapDepthtoGray(depth_image1, Converter::colorNormalize(color_image1));
    cv::Mat gray_image2 = Converter::mapDepthtoGray(depth_image2, Converter::colorNormalize(color_image2));
    assert(gray_image1.type() == CV_32FC1);


    // image window
    cv::namedWindow("gray", cv::WINDOW_NORMAL);
    cv::namedWindow("depth", cv::WINDOW_NORMAL);
    cv::namedWindow("grad", cv::WINDOW_NORMAL);

    cv::Mat1f xi(math::se3::log(cv::Mat1f::eye(4, 4)));

    // create frame pyramid
    std::vector<Frame> frames = createFramePyramid(
        5,
        Params::KINECTV2_INTRINSIC_DEPTH,
        Params::KINECTV2_INTRINSIC_RGB,
        gray_image1, depth_image1,
        gray_image2, depth_image2);

    std::vector<float> residuals;

    for (int level = 0; level < 2; level++) {
        Frame frame = frames.at(level);
        const int ROW = frame.rows;
        const int COL = frame.cols;

        std::cout << "\nLEVEL: " << level << " ROW: " << ROW << " COL: " << COL << std::endl;
        std::cout << frame.intrinsic_depth << std::endl;

        // gradient gray_image
        cv::Mat gradient_image_x = gradiate(frame.gray_image2, true);
        cv::Mat gradient_image_y = gradiate(frame.gray_image2, false);


        // show image
        {
            cv::Mat show_image;
            cv::hconcat(frame.gray_image1, frame.gray_image2, show_image);
            show_image.convertTo(show_image, CV_8UC1, 255);
            cv::imshow("gray", show_image);

            cv::hconcat(frame.depth_image1, frame.depth_image2, show_image);
            show_image.convertTo(show_image, CV_8UC1, 100);
            cv::imshow("depth", show_image);

            cv::hconcat(gradient_image_x, gradient_image_y, show_image);
            show_image.convertTo(show_image, CV_8UC1, 100, 100);
            cv::imshow("grad", show_image);
            int key = cv::waitKey(0);
            if (key == 'q')
                return 0;
        }

        // iterate direct-method
        for (int iteration = 0; iteration < 30; iteration++) {
            float residual = 0;
            std::cout << "ITERATION: " << iteration << " " << std::endl;
            // A xi + B = 0
            cv::Mat1f A(cv::Mat1f::zeros(0, 6));
            cv::Mat1f B(cv::Mat1f::zeros(0, 1));

            for (int col = 0; col < COL; col += 5) {
                for (int row = 0; row < ROW; row += 5) {
                    cv::Point2f x_i0 = cv::Point2f(col, row);
                    float depth = frame.depth_image2.at<float>(x_i0);
                    if (depth < 0.001)  // 1e-6[m]
                        continue;

                    // calc residual
                    cv::Point2f x_i = warp(xi, x_i0, depth, frame);
                    if ((!isRange(x_i.x, 0, COL)) or (!isRange(x_i.y, 0, ROW)))
                        continue;

                    float a = Converter::getColorSubpix(frame.gray_image2, x_i);
                    float b = frame.gray_image1.at<float>(x_i0);
                    float r = a - b;
                    if (isEpsilon(r))
                        continue;
                    residual += r * r;

                    // calc jacobian
                    float gx = Converter::getColorSubpix(gradient_image_x, x_i);
                    float gy = Converter::getColorSubpix(gradient_image_y, x_i);
                    cv::Mat1f jacobian1 = (cv::Mat1f(1, 2) << gx, gy);
                    cv::Mat1f jacobian2 = cv::Mat1f::zeros(2, 6);

                    if (isEpsilon(jacobian1(0)) or isEpsilon(jacobian1(1)))
                        continue;
                    // std::cout << jacobian1 << std::endl;
                    {
                        cv::Point3f x_c = cv::Point3f(Converter::backProject(frame.intrinsic_depth, cv::Mat1f(x_i0), depth));
                        float x = x_c.x, y = x_c.y, z = x_c.z;
                        float fx = frame.intrinsic_depth(0, 0);
                        float fy = frame.intrinsic_depth(1, 1);
                        jacobian2(0, 0) = fx / z;
                        jacobian2(0, 1) = 0;
                        jacobian2(0, 2) = -fx * x / z / z;
                        jacobian2(1, 0) = 0;
                        jacobian2(1, 1) = fy / z;
                        jacobian2(1, 2) = -fy * y / z / z;
#ifndef DEBUG3
                        jacobian2(0, 3) = -fx * x * y / z / z;
                        jacobian2(0, 4) = fx * (1 + x * x / z / z);
                        jacobian2(0, 5) = -fx * y / z;
                        jacobian2(1, 3) = -fy * (1 + y * y / z / z);
                        jacobian2(1, 4) = fy * x * y / z / z;
                        jacobian2(1, 5) = fy * x / z;
#endif
                    }
                    // std::cout << jacobian2 << " " << r << std::endl;
                    // stack coefficient
                    cv::vconcat(A, jacobian1 * jacobian2, A);
                    cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r), B);
                }
            }

            cv::Mat1f xi_update = solveNE(A, B);
            std::cout << "xi: " << xi.t() << std::endl;
            xi = math::se3::concatenate(xi, xi_update);
            std::cout << "dxi: " << xi_update.t() << std::endl;
            std::cout << "xi : " << xi.t() << std::endl;
            std::cout << "r  : " << residual << std::endl;
            residuals.push_back(residual);
            for (int i = 0; i < 6; i++) {
                if (std::isnan(xi(i))) {
                    std::cout << "xi is nan" << std::endl;
                    abort();
                }
            }
        }
    }
    std::cout << "DONE\n "
              << math::se3::exp(xi) << std::endl;

    for (int i = 0; i < residuals.size(); i++) {
        std::cout << i << " " << residuals.at(i) << std::endl;
    }
}


// warp先の画素値を返す
cv::Point2f warp(const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth, const Frame& frame)
{
    cv::Mat1f x_c = Converter::backProject(frame.intrinsic_depth, Converter::toMat1f(x_i.x, x_i.y), depth);
    x_c = Converter::transformByXi(xi, x_c);
    cv::Mat1f transformed_x_i = Converter::project(frame.intrinsic_depth, x_c);
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
                if (position[1] + 1 >= size.width)
                    return;

                float x0 = gray_image.at<float>(position[0], position[1]);
                float x1 = gray_image.at<float>(position[0], position[1] + 1);
                if (isEpsilon(x0) or isEpsilon(x1))
                    return;
                p = x1 - x0;
            });
    } else {
        gradiate_image.forEach<float>(
            [=](float& p, const int position[2]) -> void {
                if (position[0] + 1 >= size.height)
                    return;

                float y0 = gray_image.at<float>(position[0], position[1]);
                float y1 = gray_image.at<float>(position[0] + 1, position[1]);
                if (isEpsilon(y0) or isEpsilon(y1))
                    return;

                p = y1 - y0;
            });
    }
    return gradiate_image;
}

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

        cv::pyrDown(leveled_depth_image1, leveled_depth_image1, leveled_depth_image1.size() / 2);
        cv::pyrDown(leveled_depth_image2, leveled_depth_image2, leveled_depth_image2.size() / 2);
        cv::pyrDown(leveled_gray_image1, leveled_gray_image1, leveled_gray_image1.size() / 2);
        cv::pyrDown(leveled_gray_image2, leveled_gray_image2, leveled_gray_image2.size() / 2);
    }

    return frames;
}
