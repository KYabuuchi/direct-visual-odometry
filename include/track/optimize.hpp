#pragma once
#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/transform.hpp"
#include "system/frame.hpp"

namespace Track
{
class Scene
{
public:
    Scene(const System::Frame& frame)
        : m_depth(frame.m_depth), m_gray(frame.m_gray),
          m_intrinsic(frame.m_intrinsic),
          cols(frame.cols), rows(frame.rows), id(frame.id) {}

    // use only in tracking mode
    Scene(const cv::Mat& gray_image, const cv::Mat& depth_image, const cv::Mat1f& intrinsic, int id)
        : m_depth(depth_image), m_gray(gray_image),
          m_intrinsic(intrinsic),
          cols(depth_image.cols), rows(depth_image.rows), id(-1) {}

    const int cols;
    const int rows;
    const int id;

    static std::vector<Scene> createScenePyramid(
        const System::Frame& frame,
        const int level)
    {
        std::vector<Scene> scenes;
        Scene origin = Scene(frame);
        for (int i = 0; i < level; i++) {
            scenes.push_back(downscaleScene(origin, level - i - 1));  // level-1 , ... , 1 , 0
        }
        return scenes;
    }

    static std::vector<Scene> createScenePyramid(
        const cv::Mat& gray_image,
        const cv::Mat& depth_image,
        const cv::Mat1f& intrinsic,
        const int level)
    {
        std::vector<Scene> scenes;
        Scene origin = Scene(gray_image, depth_image, intrinsic, -1);
        for (int i = 0; i < level; i++) {
            scenes.push_back(downscaleScene(origin, level - i - 1));  // level-1 , ... , 1 , 0
        }
        return scenes;
    }

    cv::Mat1f depth() const { return m_depth; }
    cv::Mat1f gray() const { return m_gray; }
    cv::Mat1f intrinsic() const { return m_intrinsic; }
    cv::Mat1f gradX()
    {
        if (m_grad_x.empty())
            m_grad_x = Convert::gradiate(m_gray, true);
        return m_grad_x;
    }
    cv::Mat1f gradY()
    {
        if (m_grad_y.empty())
            m_grad_y = Convert::gradiate(m_gray, false);
        return m_grad_y;
    }

private:
    cv::Mat1f m_depth;
    cv::Mat1f m_gray;
    cv::Mat1f m_intrinsic;
    cv::Mat1f m_grad_x;
    cv::Mat1f m_grad_y;

    static Scene downscaleScene(const Scene& scene, int times = 1)
    {
        if (times == 0)
            return Scene(scene);
        cv::Mat depth_image = Convert::cullImage(scene.depth(), times);
        cv::Mat gray_image = Convert::cullImage(scene.gray(), times);
        cv::Mat1f intrinsic = Convert::cullIntrinsic(scene.intrinsic(), times);
        return Scene(gray_image, depth_image, intrinsic, scene.id);
    }
};

struct Stuff {
    Stuff(
        Scene& pre,
        Scene& cur,
        const cv::Mat1f& xi)
        : pre_gray(pre.gray()), pre_depth(pre.depth()),
          cur_gray(cur.gray()), cur_depth(cur.depth()),
          grad_x(cur.gradX()), grad_y(cur.gradY()),
          warped_gray(Transform::warpImage(xi, cur.gray(), cur.depth(), cur.intrinsic())),
          xi(xi), intrinsic(cur.intrinsic()),
          cols(cur.cols), rows(cur.rows) {}

    void update(const cv::Mat1f _xi)
    {
        xi = _xi;
        warped_gray = Transform::warpImage(xi, cur_gray, cur_depth, intrinsic);
    }

    void show(const std::string& window_name) const
    {
        Draw::showImage(window_name, pre_gray, pre_depth, warped_gray, cur_gray, cur_depth);
    }

    const cv::Mat pre_gray;
    const cv::Mat pre_depth;
    const cv::Mat cur_gray;
    const cv::Mat cur_depth;
    const cv::Mat grad_x;
    const cv::Mat grad_y;
    cv::Mat warped_gray;
    cv::Mat1f xi;
    const cv::Mat1f intrinsic;
    const int cols;
    const int rows;
};

struct Outcome {
    Outcome(const cv::Mat1f& xi_update, float residual, int valid_pixels)
        : xi_update(xi_update), residual(residual), valid_pixels(valid_pixels) {}
    const cv::Mat1f xi_update;
    const float residual;
    const int valid_pixels;
};

Outcome optimize(const Stuff& stuff);

}  // namespace Track