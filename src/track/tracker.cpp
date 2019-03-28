#include "track/tracker.hpp"
#include "core/convert.hpp"
#include "core/math.hpp"
#include "core/transform.hpp"
#include "track/optimize.hpp"

namespace Track
{
cv::Mat1f Tracker::track(const cv::Mat& depth_image, const cv::Mat& gray_image)
{
    m_cur_frames = createFramePyramid(depth_image, gray_image, m_config.intrinsic, m_config.level);
    cv::Mat1f xi = math::se3::xi({0, 0, 0, 0, 0, 0});

    if (m_initialized == false) {
        m_initialized = true;
        m_pre_frames = std::move(m_cur_frames);
        return math::se3::exp(xi);
    }

    for (int level = 0; level < 5 - 1; level++) {
        const Frame& pre_frame = m_pre_frames.at(level);
        const Frame& cur_frame = m_cur_frames.at(level);
        const int COLS = pre_frame.cols;
        const int ROWS = pre_frame.rows;

        // TODO: 本当は勾配計算はここで1回でいい

        if (m_config.is_chatty)
            std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        for (int iteration = 0; iteration < 10; iteration++) {
            Scene scene = {pre_frame, cur_frame, xi};
            Outcome outcome = optimize(scene);

            cv::Mat1f updated_xi = math::se3::concatenate(xi, outcome.xi_update);
            if (math::testXi(updated_xi))
                xi = updated_xi;

            if (m_config.is_chatty)
                std::cout << "iteration: " << iteration
                          << " r: " << outcome.residual
                          << " update: " << cv::norm(outcome.xi_update)
                          << " xi: " << xi.t() << std::endl;

            const std::string name = "show";
            scene.show(name);
            cv::waitKey(50);

            if (cv::norm(outcome.xi_update) < m_config.minimum_update
                or outcome.residual < m_config.minimum_residual)
                break;
        }
    }

    m_pre_frames = std::move(m_cur_frames);
    return math::se3::exp(xi);
}

}  // namespace Track