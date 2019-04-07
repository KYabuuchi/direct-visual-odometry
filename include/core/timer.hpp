#pragma once
#include <chrono>
#include <iostream>

class Timer
{
public:
    Timer(const std::string& id = "")
        : m_id(id), m_raii(id != "")
    {
        m_start = std::chrono::system_clock::now();
    }
    ~Timer()
    {
        if (m_raii) {
            long count = millSeconds();
            std::cout << m_id << ": " << count << " [ms]" << std::endl;
        }
    }
    void reset()
    {
        m_start = std::chrono::system_clock::now();
    }
    long millSeconds()
    {
        auto dur = std::chrono::system_clock::now() - m_start;
        return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    }

private:
    std::chrono::system_clock::time_point m_start;
    const std::string m_id;
    const bool m_raii;
};