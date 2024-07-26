#pragma once
#include "Model.h"
#include <fstream>
#include <print>
#include "BYTETracker.h"

class Yolov10Trace:public yo::Model{

struct Params_trace{
    int camera_fps = 60;
    int buffer_size = 30;
    float score = 0.5f;
    float nms = 0.5f;
};

private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;
    std::unique_ptr<BYTETracker> tracker; // 多目标跟踪
    
    Params_trace parms;
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
    ~Yolov10Trace(){if(session != nullptr) delete session;};
    int setparms(Params_trace parms);
    int initialize(std::string onnx_path, bool is_cuda);
    int inference(cv::Mat &image);
    
};