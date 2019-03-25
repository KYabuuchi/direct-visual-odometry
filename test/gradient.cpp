#include "calibration/loader.hpp"
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
    Calibration::Loader config_loader("../camera-calibration/data/kinectv2_00/config.yaml");
    Params::init(config_loader.rgb(), config_loader.depth(), config_loader.extrinsic());

    std::cout << "'q': quit, 'other': load next image\n"
              << std::endl;

    int num = 0;
    while (true) {
        cv::Mat rgb_image, depth_image;
        bool flag = image_loader.getNormalizedUndistortedImages(num, rgb_image, depth_image);
        if (flag == false)
            return 0;

        // map Depth->Color
        cv::Mat mapped_image = Transform::mapDepthtoGray(depth_image, rgb_image);
        cv::Mat grad_x_image = Convert::gradiate(mapped_image, true);
        cv::Mat grad_y_image = Convert::gradiate(mapped_image, false);

        cv::Mat show_image;
        cv::hconcat(Draw::visiblizeGrayImage(mapped_image), Draw::visiblizeGradientImage(grad_x_image, grad_y_image), show_image);
        cv::imshow("grad", show_image);

        num++;
        if (cv::waitKey(0) == 'q')
            break;
    }
}
