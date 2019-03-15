#include <iostream>
#include <opencv2/opencv.hpp>

namespace io
{
class Loader
{
private:
    std::vector<std::string> m_file_paths1;
    std::vector<std::string> m_file_paths2;
    const std::string m_file_paths_file;

public:
    Loader(const std::string& file_paths_file) : m_file_paths_file(file_paths_file)
    {
        std::ifstream ifs(file_paths_file);
        if (not ifs.is_open()) {
            std::cout << "[ERROR] can not open " << file_paths_file << std::endl;
            abort();
        }

        std::string dir = directorize(file_paths_file);

        while (not ifs.eof()) {
            std::string dual_file_path;
            std::getline(ifs, dual_file_path);
            if (not dual_file_path.empty()) {
                std::istringstream stream(dual_file_path);
                std::string file_path;
                getline(stream, file_path, ' ');
                m_file_paths1.push_back(dir + file_path);
                getline(stream, file_path, ' ');
                m_file_paths2.push_back(dir + file_path);
            }
        }
        ifs.close();
    }

    // 画像を取得
    bool readImages(size_t num, cv::Mat& image1, cv::Mat& image2)
    {
        if (num >= m_file_paths1.size())
            return false;

        image1 = cv::imread(m_file_paths1.at(num), cv::IMREAD_UNCHANGED);
        image2 = cv::imread(m_file_paths2.at(num), cv::IMREAD_UNCHANGED);
        return true;
    }

    // ファイル名部分を消す
    std::string directorize(std::string file_path)
    {
        while (true) {
            if (file_path.empty() or *(file_path.end() - 1) == '/')
                break;
            file_path.erase(file_path.end() - 1);
        }
        return file_path;
    }
};

}  // namespace io