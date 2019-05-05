#include "track/tracker.hpp"
#include "core/convert.hpp"
#include "core/timer.hpp"
#include "core/transform.hpp"
#include "math/math.hpp"
#include "track/optimize.hpp"
#include <chrono>

namespace Track
{
namespace
{
constexpr bool CHATTY = true;
// NOTE: MapperのDebugのために厳しい制約にしている．
constexpr float MINIMUM_RESIDUAL = 0.010f;
constexpr float MINIMUM_UPDATE = 1.e-4f;
constexpr int MAXIMUM_TIME_MS = 50;
constexpr int MAXIMUM_ITERATION = 15;
}  // namespace

cv::Mat1f Tracker::track(
    const pFrame obj_frame,
    const pFrame ref_frame)
{
    std::cout << "Tracker::track" << std::endl;

    cv::Mat1f xi = math::se3::xi();
    for (int level = 0; level < ref_frame->levels; level++) {
        const pScene obj_scene = obj_frame->at(level);
        const pScene ref_scene = ref_frame->at(level);
        const int COLS = ref_scene->cols;
        const int ROWS = ref_scene->rows;

        if (CHATTY)
            std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        Stuff stuff = {obj_scene, ref_scene, xi};

        for (int iteration = 0; iteration < MAXIMUM_ITERATION; iteration++) {
            Timer timer;
            Outcome outcome = optimize(stuff);

            cv::Mat1f updated_xi = math::se3::concatenate(xi, outcome.xi_update);
            if (math::testXi(updated_xi)) {
                xi = updated_xi;
            } else {
                std::cout << "ERROR: invalid xi_udpate" << outcome.xi_update.t() << std::endl;
            }
            stuff.update(xi);

            long mili_sec = timer.millSeconds();
            // if (CHATTY)
            //     std::cout << "itr: " << iteration
            //               << " r: " << outcome.residual
            //               << " upd: " << cv::norm(outcome.xi_update)
            //               << " rows : " << outcome.valid_pixels
            //               << " time: " << mili_sec << " ms" << std::endl;

            stuff.show(window_name);
            cv::waitKey(1);

            if (cv::norm(outcome.xi_update) < MINIMUM_UPDATE
                or outcome.residual < MINIMUM_RESIDUAL
                or mili_sec > MAXIMUM_TIME_MS)
                break;
            // TODO: 十分にupdateとresidualが小さければreturnする
        }
    }

    return xi;
}

}  // namespace Track