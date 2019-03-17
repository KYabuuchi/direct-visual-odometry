#include "core/params.hpp"

namespace Params
{
Calibration::IntrinsicParams rgb_intrinsic;
Calibration::IntrinsicParams depth_intrinsic;
Calibration::ExtrinsicParams extrinsic;

void init(
    Calibration::IntrinsicParams rgb,
    Calibration::IntrinsicParams depth,
    Calibration::ExtrinsicParams ext)
{
    rgb_intrinsic = rgb;
    depth_intrinsic = depth;
    extrinsic = ext;
}

}  // namespace Params