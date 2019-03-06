#include "math.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    using namespace math;
    {
        cv::Mat1f mat = cv::Mat1f::eye(4, 4);
        std::cout << mat << std::endl;

        mat = se3::log(mat);
        std::cout << mat << std::endl;

        mat = se3::exp(mat);
        std::cout << mat << std::endl;
    }
    std::cout << "==== test1 done ====\n"
              << std::endl;

    {
        cv::Mat1f mat(6, 1);
        cv::randu(mat, -10.0f, 10.0f);
        std::cout << mat << std::endl;
        mat = se3::exp(mat);
        std::cout << mat << std::endl;
        mat = se3::log(mat);
        std::cout << mat << std::endl;
        mat = se3::exp(mat);
        std::cout << mat << std::endl;
    }
    std::cout << "==== test2 done ====\n"
              << std::endl;
}