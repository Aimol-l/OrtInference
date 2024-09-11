#pragma once
#include "Model.h"
#include <fstream>
#include <print>

class Yolov10SAM:public yo::Model{

struct ParamsSam{
    float score = 0.5f;
    float nms = 0.5f;
};

private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;
    ParamsSam parms;

    //Env
    Ort::Env yolo_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"yolov10sam");
	Ort::Env encoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"sam_encoder");
	Ort::Env decoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"sam_decoder");
	
	//onnx会话配置相关
	Ort::Session* yolo_session = nullptr;
	Ort::Session* encoder_session = nullptr;
	Ort::Session* decoder_session = nullptr;
	
	//输入相关
	std::vector<yo::Node> yolo_input_nodes;
	std::vector<yo::Node> encoder_input_nodes;
	std::vector<yo::Node> decoder_input_nodes;
	//输出相关
	std::vector<yo::Node> yolo_output_nodes;
	std::vector<yo::Node> encoder_output_nodes;
	std::vector<yo::Node> decoder_output_nodes;

	std::vector<cv::Mat> input_images;

    //options
	Ort::SessionOptions yolo_options = Ort::SessionOptions();
	Ort::SessionOptions encoder_options = Ort::SessionOptions();
	Ort::SessionOptions decoder_options = Ort::SessionOptions();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image);
    void postprocess(std::vector<Ort::Value>& output_tensors);
    std::vector<cv::Rect> yolo_infer(std::vector<Ort::Value>&);
	std::vector<Ort::Value> encoder_infer(std::vector<Ort::Value>&);
	std::vector<Ort::Value> decoder_infer(std::vector<Ort::Value>&);
public:
    Yolov10SAM(){};
	Yolov10SAM(const Yolov10SAM&) = delete;// 删除拷贝构造函数
    Yolov10SAM& operator=(const Yolov10SAM&) = delete;// 删除赋值运算符
    ~Yolov10SAM(){
        if (yolo_session != nullptr) delete yolo_session;
		if (encoder_session != nullptr) delete encoder_session;
		if (decoder_session != nullptr) delete decoder_session;
    };
    int setparms(ParamsSam parms);
    std::variant<bool,std::string> initialize(std::vector<std::string>& onnx_paths, bool is_cuda)override;
    std::variant<bool,std::string> inference(cv::Mat &image)override;
    
};