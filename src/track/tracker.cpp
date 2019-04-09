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
    cv::Mat1f xi = math::se3::xi();
    for (int level = 0; level < m_config.level; level++) {
        const pScene ref_scene = ref_frame->at(level);
        const pScene obj_scene = cur_frame->at(level);
        const int COLS = ref_scene->cols;
        const int ROWS = ref_scene->rows;

        if (m_config.is_chatty)
            std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        Stuff stuff = {obj_scene, ref_scene, xi};
        for (int iteration = 0; iteration < 15; iteration++) {
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

    return xi;
}

}  // namespace Track