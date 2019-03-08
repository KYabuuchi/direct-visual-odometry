// math::se3のテスト
#include "math.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

bool loop = true;
int main()
{
    using namespace math;
    // viz準備
    cv::viz::Viz3d viz_window("3D-VIEW");
    viz_window.showWidget("coordinate", cv::viz::WCoordinateSystem(0.5));
    viz_window.registerKeyboardCallback([](const cv::viz::KeyboardEvent&, void*) -> void { loop = false; }, &viz_window);

    // randomな回転行列の生成
    cv::Mat1f seed(6, 1);
    cv::randu(seed, 1.0f, 3.0f);
    cv::Mat1f T = se3::exp(seed);
    cv::Mat1f xi = se3::log(T);

    // 座標変換
    cv::Point3f origin(0, 0, 0);
    cv::Mat1f before = (cv::Mat1f(4, 1) << 0, 0, 1, 1);
    cv::Mat1f after(T * before);

    // 移動前(黒)，移動後(白)
    cv::viz::WArrow arrow_b(origin, cv::Point3f(before.rowRange(0, 3)), 0.01, cv::viz::Color::black());
    cv::viz::WArrow arrow_a(cv::Point3f(T.col(3).rowRange(0, 3)), cv::Point3f(after.rowRange(0, 3)), 0.01, cv::viz::Color::white());
    viz_window.showWidget("before", arrow_b);
    viz_window.showWidget("after", arrow_a);

    // 補完(黄)
    for (int i = 1; i < 10; i++) {
        cv::Mat1f tmp = se3::exp(cv::Mat1f(xi * static_cast<float>(i) / 10.0));
        cv::Mat1f inter(tmp * before);
        cv::viz::WArrow arrow_i(cv::Point3f(tmp.col(3).rowRange(0, 3)), cv::Point3f(inter.rowRange(0, 3)), 0.01, cv::viz::Color::yellow());
        viz_window.showWidget("inter" + std::to_string(i), arrow_i);
    }
    // 予測(緑)
    for (int i = 1; i < 10; i++) {
        cv::Mat1f tmp = se3::exp(cv::Mat1f(xi * static_cast<float>(10.0 + i) / 10.0));
        cv::Mat1f predict(tmp * before);
        cv::viz::WArrow arrow_p(cv::Point3f(tmp.col(3).rowRange(0, 3)), cv::Point3f(predict.rowRange(0, 3)), 0.01, cv::viz::Color::lime());
        viz_window.showWidget("predict" + std::to_string(i), arrow_p);
    }

    while (loop) {
        viz_window.spinOnce(1, true);
    }
}