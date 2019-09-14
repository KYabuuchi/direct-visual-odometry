#include "calibration/loader.hpp"
#include "core/loader.hpp"
#include "graphic/draw.hpp"
#include "system/system.hpp"
#include <Eigen/Dense>

const std::string window_name = "trajectry";
void show(const std::list<cv::Mat1f>& trajectory);

//#define USE_CAMERA

int main(int argc, char* argv[])
{
    Graphic::initialize();

    std::string input_file = "../data/logicool0/info.txt";
    if (argc == 2)
        input_file = argv[1];

    cv::namedWindow("Source", cv::WINDOW_NORMAL);

    Core::Loader loader(input_file, "../external/camera-calibration/data/logicool_00/config.yaml");
    std::cout << "original internal parameters\n"
              << loader.Rgb().K() << std::endl;

#ifdef USE_CAMERA
    cv::VideoCapture video("/dev/video1");
#else
    int num = 0;
#endif

    // main system
    System::VisualOdometry vo(loader.Rgb().K());
    std::list<cv::Mat1f> trajectory;

    while (true) {
        cv::Mat1f gray_image;
#ifdef USE_CAMERA
        cv::Mat color_image, raw_image;
        video >> raw_image;
        cv::cvtColor(raw_image, color_image, cv::COLOR_BGR2GRAY);
        color_image.convertTo(gray_image, CV_32FC1, 1.0 / 255.0);
#else
        if (not loader.getNormalizedUndistortedImages(num++, gray_image))
            break;
#endif

        // odometrize
        cv::Mat1f T = vo.odometrize(gray_image);
        T = Convert::inversePose(T);
        trajectory.push_back(T);
        if (trajectory.size() > 50)
            trajectory.pop_front();
        show(trajectory);

        // wait
        int key = cv::waitKey(10);
        if (key == 'q')
            break;
        if (key == 's') {
            while (cv::waitKey(0) != 'r')
                ;
        }

#ifdef USE_CAMERA
        cv::imshow("Source", raw_image);
#endif

        if (!Graphic::isRunning())
            break;
    }

    Graphic::finalize();
}

void show(const std::list<cv::Mat1f>& trajectory)
{
    Graphic::clear();
    std::vector<Eigen::Vector2d> tmp;
    tmp.reserve(trajectory.size());
    for (const cv::Mat1f& e : trajectory) {
        Eigen::Vector2d v = 10 * Eigen::Vector2d(e(0, 3), e(2, 3));
        tmp.push_back(v);
    }
    Graphic::draw(tmp, Graphic::Form::CURVE, Graphic::Color::YELLOW);
}
