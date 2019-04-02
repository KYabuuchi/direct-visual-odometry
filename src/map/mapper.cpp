#include "map/mapper.hpp"


namespace Map
{

bool Mapper::insertableHistory(pFrame new_frame)
{
    cv::Mat1f& xi = new_frame->m_relative_xi;
    double scalar = cv::norm(xi.rowRange(0, 3));

    return scalar > 0.10;  // 100[mm]
}

void Mapper::initializeHistory(FrameHistory frame_history, pFrame new_frame)
{
}
void Mapper::propagate(FrameHistory frame_history, pFrame new_frame)
{
}
void Mapper::update(FrameHistory frame_history, pFrame new_frame)
{
}
void Mapper::regularize(FrameHistory frame_history)
{
}


}  // namespace Map
