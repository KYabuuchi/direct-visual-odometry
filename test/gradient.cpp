#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/loader.hpp"
#include "core/transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::namedWindow("grad", CV_WINDOW_NORMAL);
    cv::resizeWindow("grad", 960, 720);

    Loader image_loader("../data/KINECT_1DEG/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");

    std::cout << "'q': quit, 'other': load next image\n"
              << std::endl;

    int num = 0;
    while (true) {
        cv::Mat1f gray_image, depth_image, sigma_image;
        bool flag = image_loader.getMappedImages(num, gray_image, depth_image, sigma_image);
        if (flag == false)
            return 0;

        // map Depth->Color
        cv::Mat1f grad_x_image = Convert::gradiate(gray_image, true);
        cv::Mat1f grad_y_image = Convert::gradiate(gray_image, false);

        cv::Mat show_image;
        cv::hconcat(
            Draw::visualizeGray(gray_image),
            Draw::visualizeGradient(grad_x_image, grad_y_image),
            show_image);
        cv::imshow("grad", show_image);

        num++;
        if (cv::waitKey(0) == 'q')
            break;
    }
}
