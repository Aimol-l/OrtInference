#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <random>
#include <chrono>
#include <format>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <type_traits>
#include <opencv2/opencv.hpp>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <libgen.h>
    #include <unistd.h>
#endif


namespace MT{
    // 获取文件名
    inline std::string GetFileName(const std::string& path){
        size_t pos = path.find_last_of("/\\");
        return (pos == std::string::npos) ? path : path.substr(pos + 1);
    }
    // 获取文件后缀
    inline std::string GetFileExten(const std::string& path) {
        size_t pos = path.rfind('.');
        return (pos == std::string::npos) ? "" : path.substr(pos + 1);
    }
    // 相对路径转为绝对路径
    inline std::string GetFileAbsPath(const std::string& path){
        std::string absPath;
        #ifdef _WIN32
            // Windows平台使用GetFullPathName
            char buffer[1024];
            DWORD result = GetFullPathName(path.c_str(), 1024, buffer, nullptr);
            if (result > 0 && result < 1024) {
                absPath = buffer;
            } else {
                // 处理错误情况
                std::cerr << "GetFullPathName failed" << std::endl;
            }
        #else
            // POSIX平台使用realpath
            char buffer[1024];
            if (realpath(path.c_str(), buffer) != nullptr) {
                absPath = buffer;
            } else {
                throw std::runtime_error("realpath failed");
            }
        #endif
        return absPath;
    }
    // 文件是否存在
    inline bool FileExists(std::string& path){
        std::ifstream file(path.c_str());
        return file.good();
    }
    inline bool FileExists(const char* path){
        std::ifstream file(path);
        return file.good();
    }
    // 分割字符串
    inline std::vector<std::string> SplitString(const std::string& src,const char gap = ' '){
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream iss(src);
        while (std::getline(iss, token, gap)) {
            tokens.push_back(token);
        }
        return std::move(tokens);
    }
    // 合并字符串
    inline std::string JoinStrings(const std::vector<std::string>& strings,const char gap = ' '){
        std::string result;
        for(const auto &str:strings){
            result = result + std::string{gap} + str;
        }
        return result;
    }
    // 字符串是否以start开头
    inline bool StartsWith(const std::string& src, const std::string& start) {return src.compare(0, start.size(), start) == 0;}
    // 字符串是否以end结尾
    inline bool EndsWith(const std::string& src, const std::string& end) {return src.compare(src.size() - end.size(), end.size(), end) == 0;}
    //替换文件后缀
    inline std::string ReplaceExten(const std::string& src,const std::string& exten){
        size_t pos = src.rfind('.');
        if(pos == std::string::npos){
            return std::format("{}{}",src,exten);
        }
        return std::format("{}{}",src.substr(0,pos),exten);
    }
    // 获取一个[min,max]范围内的随机浮点数字，均匀采样
    template<typename T>
    inline T GetRandom(T min, T max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<T> dis(min, max);
            return dis(gen);
        } else if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> dis(min, max);
            return dis(gen);
        } else {
            static_assert(std::is_arithmetic<T>::value, "GetRandom only supports arithmetic types");
        }
    }
    // 获取当前时间字符串
    inline std::string GetCurrentTime(){
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm buf;
    #if defined(__unix__) || defined(__APPLE__)
        localtime_r(&in_time_t, &buf);
    #elif defined(_MSC_VER)
        localtime_s(&buf, &in_time_t);
    #else
        // Other platforms not supported
        static_assert(false, "Platform not supported");
    #endif
        std::ostringstream ss;
        ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S"); // Format: YYYY-MM-DD HH:MM:SS
        return ss.str();
    }
    // 获取当前日期
    inline std::string GetCurrentDate() {
        // 获取当前时间点
        auto now = std::chrono::system_clock::now();
        // 转换为 time_t 类型
        std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
        // 转换为 tm 结构
        std::tm localTime = *std::localtime(&timeNow);
        // 格式化为字符串，只保留日期部分
        std::ostringstream oss;
        oss << std::put_time(&localTime, "%Y-%m-%d");
        return oss.str();
    }
    // 获取当前时间戳,按毫秒算
    inline uint64_t GetCurrentTimestamp(){
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        return static_cast<uint64_t>(milliseconds.count());
    }
    inline size_t GetStrNumber(const std::string& str) {
        size_t count = 0;
        for (size_t i = 0; i < str.size(); ) {
            unsigned char c = str[i];
            if ((c & 0x80) == 0)  // 单字节字符 (ASCII)
                ++i;
            else if ((c & 0xE0) == 0xC0) // 双字节字符
                i += 2;
            else if ((c & 0xF0) == 0xE0) // 三字节字符
                i += 3;
            else if ((c & 0xF8) == 0xF0) // 四字节字符
                i += 4;
            else 
                ++i;
            ++count;
        }
        return count;
    }
    // 根据逻辑字符的起始和结束位置获取子字符串
    inline std::string GetSubStr(const std::string str, size_t start, size_t stop) {
        if (start > stop || stop > GetStrNumber(str)) 
            throw std::out_of_range("Invalid start or stop index");
        size_t current_char_index = 0;
        size_t start_byte_index = 0;
        size_t end_byte_index = str.size();
        // 找到起始字节索引
        for (size_t i = 0; i < str.size(); ) {
            if (current_char_index == start) {
                start_byte_index = i;
                break;
            }
            unsigned char c = static_cast<unsigned char>(str[i]);
            if ((c & 0x80) == 0) { // 单字节字符 (ASCII)
                ++i;
            }
            else if ((c & 0xE0) == 0xC0) { // 双字节字符
                i += 2;
            }
            else if ((c & 0xF0) == 0xE0) { // 三字节字符
                i += 3;
            }
            else if ((c & 0xF8) == 0xF0) { // 四字节字符
                i += 4;
            }
            else { // 非法UTF-8字符
                ++i;
            }
            ++current_char_index;
        }
        // 找到结束字节索引
        current_char_index = 0;
        for (size_t i = 0; i < str.size(); ) {
            if (current_char_index == stop) {
                end_byte_index = i;
                break;
            }
            unsigned char c = static_cast<unsigned char>(str[i]);
            if ((c & 0x80) == 0) { // 单字节字符 (ASCII)
                ++i;
            }
            else if ((c & 0xE0) == 0xC0) { // 双字节字符
                i += 2;
            }
            else if ((c & 0xF0) == 0xE0) { // 三字节字符
                i += 3;
            }
            else if ((c & 0xF8) == 0xF0) { // 四字节字符
                i += 4;
            }
            else { // 非法UTF-8字符
                ++i;
            }
            ++current_char_index;
        }
        return str.substr(start_byte_index, end_byte_index - start_byte_index);
    }

    //************************************************************************
    // 图片预处理:将 图片填充至指定大小，并缩放至指定大小
    inline cv::Mat PaddingImg(cv::Mat& img,cv::Size target){
        int top=0,bottom=0,left=0,right=0;
        auto ori_w = (float)img.cols;
        auto ori_h = (float)img.rows;
        auto net_w = (float)target.width;
        auto net_h = (float)target.height;
        auto r = std::min(net_w/ori_w,net_h/ori_h);//缩放比例
        auto width = int(ori_w*r);
        auto height = int(ori_h*r);
        top = (net_h - height) / 2;
        bottom = (net_h - height) / 2 + int(net_h - height) % 2;
        left = (net_w - width) / 2;
        right = (net_w - width) / 2 + int(net_w - width) % 2;
        cv::Mat result_img;
        cv::resize(img,result_img,cv::Size(width,height));
        cv::copyMakeBorder(result_img, result_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
        return result_img;
    }
    inline cv::Mat UnpaddingImg(cv::Mat& img,cv::Size target,cv::Size ori_size){
        auto ori_w = (float)ori_size.width;
        auto ori_h = (float)ori_size.height;
        auto net_w = (float)target.width;
        auto net_h = (float)target.height;
        auto r = std::min(net_w/ori_w,net_h/ori_h);//缩放比例
        auto width = int(ori_w*r);
        auto height = int(ori_h*r);
        int top = (net_h - height) / 2;
        int left = (net_w - width) / 2;
        cv::Mat result_img = img(cv::Rect(left, top, width, height)).clone();
        cv::resize(result_img,result_img,ori_size);
        std::println("[{},{},{},{}]",top,left,width,height);
        return result_img;
    }
    inline float ComputeIou(const cv::Rect& a, const cv::Rect& b) {
        // 检查输入有效性（可选）
        assert(a.width >= 0 && a.height >= 0 && b.width >= 0 && b.height >= 0);
        // 计算交集区域
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        int interWidth = std::max(0, x2 - x1);
        int interHeight = std::max(0, y2 - y1);
        // 计算面积（避免整数溢出）
        float intersection = static_cast<float>(interWidth) * interHeight;
        float areaA = static_cast<float>(a.width) * a.height;
        float areaB = static_cast<float>(b.width) * b.height;
        float unionArea = areaA + areaB - intersection;
        // 处理除零情况
        return unionArea > 0.0f ? intersection / unionArea : 0.0f;
    }
    
    //************************************************************************
    class OCRDictionary {
    public:
        OCRDictionary(){}
        OCRDictionary(const std::string& dict_path) {
            this->load(dict_path);
        }
        OCRDictionary(const char* dict_path) {
            std::string dict_path_str(dict_path);
            this->load(dict_path_str);
        }
        // 加载字典
        bool load(const std::string& dict_path) {
            std::ifstream file(dict_path);
            if (!file.is_open()) {
                std::cerr << "无法打开字典文件: " << dict_path << std::endl;
                return false;
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    dictionary_.push_back(line);
                    char_to_index_[line] = dictionary_.size() - 1;
                }
            }
            file.close();
            // std::cout << "字典加载完成，字符数量: " << dictionary_.size() << std::endl;
            return true;
        }
        // 通过索引获取字符
        std::string get_char(int index) const {
            if (index < 0 || index >= dictionary_.size()) {
                return " "; // 返回空字符串表示索引无效
            }
            return dictionary_[index];
        }

        // 通过字符获取索引
        int get_index(const std::string& ch) const {
            auto it = char_to_index_.find(ch);
            if (it != char_to_index_.end()) {
                return it->second;
            }
            return -1; // -1 表示未找到
        }
        // 获取字典大小
        size_t size() const {
            return dictionary_.size();
        }
    private:
        std::vector<std::string> dictionary_;          // 字符列表
        std::unordered_map<std::string, int> char_to_index_; // 反向查找表
    };
};