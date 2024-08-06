#pragma once
#include "Model.h"
#include <fstream>
#include <print>


class GroundDino:public yo::Model{

struct Params_dino{
    float score = 0.5f;
    float nms = 0.5f;
    std::string prompt = "biscuits";
};

private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;

    Params_dino parms;
    std::vector<yo::Node> input_nodes;
    std::vector<yo::Node> output_nodes;
    std::vector<cv::Mat> input_images;

    Ort::Session* session = nullptr;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"groundingdino");
    Ort::SessionOptions session_options = Ort::SessionOptions();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image);
    void postprocess(std::vector<Ort::Value>& output_tensors);
public:
    GroundDino(){};
    GroundDino(const GroundDino&) = delete;// 删除拷贝构造函数
    GroundDino& operator=(const GroundDino&) = delete;// 删除赋值运算符
    ~GroundDino(){if(session != nullptr) delete session;};
    int setparms(Params_dino parms);
    int initialize(std::string onnx_path, bool is_cuda);
    int inference(cv::Mat &image);
    
};