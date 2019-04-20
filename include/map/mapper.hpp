#pragma once
#include "core/transform.hpp"
#include "math/math.hpp"
#include "system/frame.hpp"
#include <memory>

namespace Map
{
using Frame = System::Frame;
using FrameHistory = System::FrameHistory;
using pFrame = std::shared_ptr<System::Frame>;
using pScene = std::shared_ptr<System::Scene>;

// 本体
class Mapper
{
public:
    Mapper() {}

    // 本体
    void estimate(FrameHistory& frame_history, pFrame frame);

    // 十分移動したか否かを判定
    bool needNewFrame(pFrame frame);

    // Ref Frameを伝搬
    pFrame propagate(const FrameHistory& frame_history, const pFrame frame);

    // 深度・分散を更新
    void update(const FrameHistory& frame_history, pFrame frame);

    // 深度を拡散
    void regularize(const FrameHistory& frame_history, pFrame frame);

private:
};
}  // namespace Map
