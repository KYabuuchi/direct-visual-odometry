#include "calibration/loader.hpp"
#include "core/loader.hpp"
#include "system/system.hpp"

const std::string window_name = "trajectry";
void show(const std::vector<cv::Mat1f>& trajectory);

int main(/*int argc, char* argv[]*/)
{
    // loading
    Core::Loader loader("../data/kinectv2_01/info.txt", "../camera-calibration/data/logicool_00/config.yaml");

    // trajectory
    // cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    // cv::Mat show_image = cv::Mat::zeros(480, 640, CV_8UC3);


    // main system
    std::cout << loader.Rgb().K() << std::endl;

    System::VisualOdometry vo(loader.Rgb().K());
    std::vector<cv::Mat1f> trajectory;

    // data
    int num = 0;
    while (true) {
        cv::Mat1f gray_image;
        if (not loader.getNormalizedUndistortedImages(num++, gray_image))
            break;

        // odometrize
        cv::Mat1f T = vo.odometrize(gray_image);
        std::cout << "\n"
                  << T << "\n"
                  << std::endl;

        // trajectory.push_back(T);
        // show(trajectory);

        // wait
        if (cv::waitKey(0) == 'q')
            break;
    }
}

void show(const std::vector<cv::Mat1f>& trajectory)
{
    cv::Mat show_image = cv::Mat::zeros(480, 640, CV_8UC3);
    double gain = 1000;
    for (int i = 1; i < trajectory.size(); i++) {
        const cv::Mat1f& T0 = trajectory.at(i - 1);
        const cv::Mat1f& T1 = trajectory.at(i);
        cv::Point center0(320 + gain * T0(0, 3), 240 + gain * T0(2, 3));
        cv::Point center1(320 + gain * T1(0, 3), 240 + gain * T1(2, 3));
        cv::line(show_image, center0, center1, CV_RGB(255, 255, 0), 3);
    }
    cv::imshow(window_name, show_image);
}