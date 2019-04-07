#include "track/tracker.hpp"
#include "core/convert.hpp"
#include "core/math.hpp"
#include "core/timer.hpp"
#include "core/transform.hpp"
#include "track/optimize.hpp"
#include <chrono>

namespace Track
{
cv::Mat1f Tracker::track(
    const std::shared_ptr<System::Frame> ref_frame,
    const std::shared_ptr<System::Frame> cur_frame)
{
    std::vector<std::shared_ptr<Scene>> cur_scenes = Scene::createScenePyramid(cur_frame, m_config.level);

    // 未初期化ならreturn
    if (m_initialized == false) {
        m_initialized = true;
        m_pre_scenes = cur_scenes;
        return math::se3::xi();
    }

    // cacheの利用
    std::vector<std::shared_ptr<Scene>> ref_scenes;
    {
        bool flag = m_pre_scenes.size() > 0 && m_pre_scenes.at(0)->id == ref_frame->id;
        std::string s = (flag ? "use cache" : "dont use cache");
        if (m_config.is_chatty)
            std::cout << s << std::endl;
        if (flag) {
            ref_scenes = m_pre_scenes;
        } else {
            ref_scenes = m_pre_scenes;
        }
    }

    cv::Mat1f xi = math::se3::xi();
    for (int level = 0; level < m_config.level - 1; level++) {
        std::shared_ptr<Scene> ref_scene = ref_scenes.at(level);
        std::shared_ptr<Scene> cur_scene = cur_scenes.at(level);
        const int COLS = ref_scene->cols;
        const int ROWS = ref_scene->rows;

        if (m_config.is_chatty)
            std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        Stuff stuff = {cur_scene, ref_scene, xi};
        for (int iteration = 0; iteration < 10; iteration++) {
            Timer timer;

            Outcome outcome = optimize(stuff);

            cv::Mat1f updated_xi = math::se3::concatenate(xi, outcome.xi_update);
            if (math::testXi(updated_xi)) {
                xi = updated_xi;
            } else {
                std::cout << "ERROR: invalid xi_udpate" << outcome.xi_update.t() << std::endl;
            }
            stuff.update(xi);

            long count = timer.millSeconds();
            if (m_config.is_chatty)
                std::cout << "itr: " << iteration
                          << " r: " << outcome.residual
                          << " upd: " << cv::norm(outcome.xi_update)
                          << " rows : " << outcome.valid_pixels
                          << " time: " << count << " ms" << std::endl;

            stuff.show("show");
            cv::waitKey(1);

            if (cv::norm(outcome.xi_update) < m_config.minimum_update
                or outcome.residual < m_config.minimum_residual
                or count > 1000)
                break;
        }
    }

    m_pre_scenes = cur_scenes;

    return xi;
}

}  // namespace Track