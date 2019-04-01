#include "track/tracker.hpp"
#include "core/convert.hpp"
#include "core/math.hpp"
#include "core/transform.hpp"
#include "track/optimize.hpp"
#include <chrono>

namespace Track
{
cv::Mat1f Tracker::track(
    const std::shared_ptr<System::Frame> ref_frame,
    const std::shared_ptr<System::Frame> cur_frame)
{
    std::vector<Scene> cur_scenes = Scene::createScenePyramid(*cur_frame, m_config.level);

    // 未初期化ならreturn
    if (m_initialized == false) {
        m_initialized = true;
        m_pre_scenes = std::move(cur_scenes);
        return math::se3::T();
    }

    cv::Mat1f xi = math::se3::xi();
    std::vector<Scene> ref_scenes = m_pre_scenes;

    for (int level = 0; level < m_config.level - 1; level++) {
        Scene& pre_scene = ref_scenes.at(level);
        Scene& cur_scene = cur_scenes.at(level);
        const int COLS = pre_scene.cols;
        const int ROWS = pre_scene.rows;

        if (m_config.is_chatty)
            std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        Stuff stuff = {pre_scene, cur_scene, xi};
        for (int iteration = 0; iteration < 10; iteration++) {
            auto start = std::chrono::system_clock::now();

            Outcome outcome = optimize(stuff);

            cv::Mat1f updated_xi = math::se3::concatenate(xi, outcome.xi_update);
            if (math::testXi(updated_xi)) {
                xi = updated_xi;
            } else {
                std::cout << "ERROR: invalid xi_udpate" << outcome.xi_update.t() << std::endl;
            }
            stuff.update(xi);

            auto dur = std::chrono::system_clock::now() - start;
            int count = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
            if (m_config.is_chatty)
                std::cout << "itr: " << iteration
                          << " r: " << outcome.residual
                          << " upd: " << cv::norm(outcome.xi_update)
                          << " rows : " << outcome.valid_pixels
                          << " time: " << count << std::endl;

            stuff.show("show");
            cv::waitKey(1);

            if (cv::norm(outcome.xi_update) < m_config.minimum_update
                or outcome.residual < m_config.minimum_residual
                or count > 1000)
                break;
        }
    }
    return math::se3::exp(xi);
}

cv::Mat1f Tracker::track(const cv::Mat& gray_image, const cv::Mat& depth_image)
{
    std::vector<Scene> cur_scenes = Scene::createScenePyramid(depth_image, gray_image, m_config.intrinsic, m_config.level);
    cv::Mat1f xi = math::se3::xi();

    if (m_initialized == false) {
        m_initialized = true;
        m_pre_scenes = std::move(cur_scenes);
        return math::se3::exp(xi);
    }

    for (int level = 0; level < m_config.level - 1; level++) {
        Scene& pre_scene = m_pre_scenes.at(level);
        Scene& cur_scene = cur_scenes.at(level);
        const int COLS = pre_scene.cols;
        const int ROWS = pre_scene.rows;

        if (m_config.is_chatty)
            std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        Stuff stuff = {pre_scene, cur_scene, xi};
        for (int iteration = 0; iteration < 10; iteration++) {
            auto start = std::chrono::system_clock::now();

            Outcome outcome = optimize(stuff);

            cv::Mat1f updated_xi = math::se3::concatenate(xi, outcome.xi_update);
            if (math::testXi(updated_xi)) {
                xi = updated_xi;
            } else {
                std::cout << "ERROR: invalid xi_udpate" << outcome.xi_update.t() << std::endl;
            }
            stuff.update(xi);

            auto dur = std::chrono::system_clock::now() - start;
            int count = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
            if (m_config.is_chatty)
                std::cout << "itr: " << iteration
                          << " r: " << outcome.residual
                          << " upd: " << cv::norm(outcome.xi_update)
                          << " rows : " << outcome.valid_pixels
                          << " time: " << count << std::endl;

            stuff.show("show");
            cv::waitKey(1);

            if (cv::norm(outcome.xi_update) < m_config.minimum_update
                or outcome.residual < m_config.minimum_residual
                or count > 1000)
                break;
        }
    }

    m_pre_scenes = std::move(cur_scenes);
    return math::se3::exp(xi);
}


}  // namespace Track