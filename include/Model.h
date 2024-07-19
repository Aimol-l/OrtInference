#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

struct Node{
    std::vector<int64_t> dim = {0,0,0,0}; // batch,channel,height,width
    char* name = nullptr;
};

class Model{
public:
    virtual ~Model(){};
    virtual int inference(cv::Mat &image) = 0;
    virtual int initialize(std::string onnx_path, bool is_cuda) = 0;
protected:
    virtual void preprocess(cv::Mat &image)=0;
    virtual void postprocess(std::vector<Ort::Value>& output_tensors)=0;
};
