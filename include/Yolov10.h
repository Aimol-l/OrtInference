#pragma once
#include "Model.h"
#include <fstream>
#include <print>


class Yolov10:public Model{

struct Params_v10{
    float score = 0.5f;
    float nms = 0.5f;
};

private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;

    Params_v10 parms;
    std::vector<Node> input_nodes;
    std::vector<Node> output_nodes;
    std::vector<cv::Mat> input_images;

    Ort::Session* session = nullptr;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"yolov10");
    Ort::SessionOptions session_options = Ort::SessionOptions();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image);
    void postprocess(std::vector<Ort::Value>& output_tensors);
public:
    Yolov10(){};
    ~Yolov10(){if(session != nullptr) delete session;};
    int setparms(Params_v10 parms);
    int initialize(std::string onnx_path, bool is_cuda);
    int inference(cv::Mat &image);
    
};