#include "track/tracker.hpp"
#include "core/convert.hpp"
#include "core/math.hpp"
#include "core/transform.hpp"
#include "track/optimize.hpp"
#include <chrono>

namespace Track
{
cv::Mat1f Tracker::track(const cv::Mat& depth_image, const cv::Mat& gray_image)
{
    m_cur_scene = Scene::createFramePyramid(depth_image, gray_image, m_config.intrinsic, m_config.level);
    cv::Mat1f xi = math::se3::xi();

    if (m_initialized == false) {
        m_initialized = true;
        m_pre_scene = std::move(m_cur_scene);
        return math::se3::exp(xi);
    }

    for (int level = 0; level < m_config.level - 1; level++) {
        const Scene& pre_scene = m_pre_scene.at(level);
        const Scene& cur_scene = m_cur_scene.at(level);
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

    m_pre_scene = std::move(m_cur_scene);
    return math::se3::exp(xi);
}


}  // namespace Track