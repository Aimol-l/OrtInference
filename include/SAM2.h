#pragma once
#include "Model.h"
#include <fstream>
#include <print>

class SAM2:public yo::Model{

struct InferenceStatus{
    int32_t current_frame = 0;
    Ort::Value obj_ptr_first;
    std::vector<Ort::Value> obj_ptr_recent;

};
struct ParamsSam2{
    int32_t buffer_size = 15;
    cv::Rect prompt_box;
};
private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;
    std::vector<cv::Mat> input_images;
    ParamsSam2 parms;
    InferenceStatus infer_status;
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

    // 推理过程tensors
    std::vector<Ort::Value> img_encoder_out; // [pix_feat,high_res_feat0,high_res_feat1,vision_feats,vision_pos_embed]
    std::vector<Ort::Value> img_decoder_out; // [obj_ptr,mask_for_mem,pred_mask]
    std::vector<Ort::Value> mem_attention_out;//[image_embed]
    std::vector<Ort::Value> mem_encoder_out; // [maskmem_features,]
protected:
    void preprocess(cv::Mat &image) override;
    void postprocess(std::vector<Ort::Value>& output_tensors) override;
    std::vector<std::string> str_split(const std::string& str, char delimiter);

    std::vector<Ort::Value> build_mem_attention_input();

    void img_encoder_infer(std::vector<Ort::Value>&);
    void img_decoder_infer();
    void mem_attention_infer();
    void mem_encoder_infer();
public:
    SAM2();
    ~SAM2();
    int initialize(std::string onnx_path, bool is_cuda) override ;
    int inference(cv::Mat &image) override;
};