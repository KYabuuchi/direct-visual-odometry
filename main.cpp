#include "calibration/loader.hpp"
#include "core/loader.hpp"
#include "graphic/draw.hpp"
#include "system/system.hpp"
#include <Eigen/Dense>

const std::string window_name = "trajectry";
void show(const std::vector<cv::Mat1f>& trajectory);

int main(/*int argc, char* argv[]*/)
{
    Graphic::initialize();

    // loading
    Core::Loader loader("../data/logicool0/info.txt", "../external/camera-calibration/data/logicool_00/config.yaml");

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

        trajectory.push_back(T);
        show(trajectory);

        // wait
        if (cv::waitKey(0) == 'q')
            break;

        if (not Graphic::isRunning())
            break;
    }

    Graphic::finalize();
}

void show(const std::vector<cv::Mat1f>& trajectory)
{
    Graphic::clear();
    std::vector<Eigen::Vector2d> tmp;
    tmp.reserve(trajectory.size());
    for (const cv::Mat1f& e : trajectory) {
        Eigen::Vector2d v = 10 * Eigen::Vector2d(e(0, 3), e(2, 3));
        std::cout << v.transpose() << std::endl;
        tmp.push_back(v);
    }

    Graphic::draw(tmp, Graphic::Form::CURVE, Graphic::Color::YELLOW);
}