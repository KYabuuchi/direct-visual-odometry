#include "calibration/loader.hpp"
#include "core/loader.hpp"
#include "graphic/draw.hpp"
#include "system/system.hpp"
#include <Eigen/Dense>

const std::string window_name = "trajectry";
void show(const std::vector<cv::Mat1f>& trajectory);

int main(int argc, char* argv[])
{
    Graphic::initialize();

    std::string input_file = "../data/logicool0/info.txt";
    if (argc == 2)
        input_file = argv[1];

    // loading
    Core::Loader loader(input_file, "../external/camera-calibration/data/logicool_00/config.yaml");

    // main system
    std::cout << "original internal parameters\n"
              << loader.Rgb().K() << std::endl;

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
        T = Convert::inversePose(T);
        trajectory.push_back(T);
        show(trajectory);

        // wait
        if (cv::waitKey(0) == 'q')
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
        tmp.push_back(v);
    }
    Graphic::draw(tmp, Graphic::Form::CURVE, Graphic::Color::YELLOW);
    const cv::Mat1f& latest = trajectory.back();
    Eigen::Vector2d bottom(10 * latest(0, 3), 10 * latest(2, 3));
    Eigen::Vector2d top(10 * latest(0, 3), 10 * latest(2, 3) + 0.1);
    tmp.clear();
    tmp.push_back(top);
    tmp.push_back(bottom);
    Graphic::drawArrow(tmp, Graphic::Color::RED);
}