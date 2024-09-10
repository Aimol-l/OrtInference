#pragma once
#include "Model.h"
#include <fstream>
#include <print>
#include "BYTETracker.h"

class Yolov10Trace:public yo::Model{

struct ParamsTrace{
    int camera_fps = 60;
    int buffer_size = 30;
    float score = 0.5f;
    float nms = 0.5f;
};

private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;
    std::unique_ptr<BYTETracker> tracker; // 多目标跟踪
    
    ParamsTrace parms;
    std::vector<yo::Node> input_nodes;
    std::vector<yo::Node> output_nodes;
    std::vector<cv::Mat> input_images;

    Ort::Session* session = nullptr;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"yolov10trace");
    Ort::SessionOptions session_options = Ort::SessionOptions();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image);
    void postprocess(std::vector<Ort::Value>& output_tensors);
public:
    Yolov10Trace(){};
    Yolov10Trace(const Yolov10Trace&) = delete;// 删除拷贝构造函数
    Yolov10Trace& operator=(const Yolov10Trace&) = delete;// 删除赋值运算符
    ~Yolov10Trace(){if(session != nullptr) delete session;};
    int setparms(ParamsTrace parms);
    int initialize(std::vector<std::string>& onnx_paths, bool is_cuda);
    int inference(cv::Mat &image);
    
};