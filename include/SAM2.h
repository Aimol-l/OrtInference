#pragma once
#include "Model.h"
#include <fstream>
#include <print>


class SAM2:public yo::Model{
private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;
    std::vector<cv::Mat> input_images;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);

    //Env
    Ort::Env img_encoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"img_encoder");
	Ort::Env img_decoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"img_decoder");
	Ort::Env mem_attention_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"mem_attention");
    Ort::Env mem_encoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"mem_encoder");

    //onnx会话配置相关
	Ort::Session* img_encoder_session = nullptr;
	Ort::Session* img_decoder_session = nullptr;
	Ort::Session* mem_attention_session = nullptr;
    Ort::Session* mem_encoder_session = nullptr;

    //options
	Ort::SessionOptions img_encoder_options = Ort::SessionOptions();
	Ort::SessionOptions img_decoder_options = Ort::SessionOptions();
	Ort::SessionOptions mem_attention_options = Ort::SessionOptions();
    Ort::SessionOptions mem_encoder_options = Ort::SessionOptions();
    
    //输入相关
	std::vector<yo::Node> img_encoder_input_nodes;
	std::vector<yo::Node> img_decoder_input_nodes;
	std::vector<yo::Node> mem_attention_input_nodes;
    std::vector<yo::Node> mem_encoder_input_nodes;

	//输出相关
	std::vector<yo::Node> img_encoder_output_nodes;
	std::vector<yo::Node> img_decoder_output_nodes;
	std::vector<yo::Node> mem_attention_output_nodes;
    std::vector<yo::Node> mem_encoder_output_nodes;
protected:
    void preprocess(cv::Mat &image) override;
    void postprocess(std::vector<Ort::Value>& output_tensors) override;
    std::vector<std::string> str_split(const std::string& str, char delimiter);

    std::vector<Ort::Value> img_encoder_infer(std::vector<Ort::Value>&);
    std::vector<Ort::Value> img_decoder_infer(std::vector<Ort::Value>&);
    std::vector<Ort::Value> mem_attention_infer(std::vector<Ort::Value>&);
    std::vector<Ort::Value> mem_encoder_infer(std::vector<Ort::Value>&);
public:
    SAM2();
    ~SAM2();
    int initialize(std::string onnx_path, bool is_cuda) override ;
    int inference(cv::Mat &image) override;
};