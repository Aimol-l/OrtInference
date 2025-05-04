#pragma once
#include "Model.h"
#include "clipper.h"
#include <fstream>
#include <print>
#include <algorithm>

class PaddleOCR:public yo::Model{

struct ParamsOCR{
    bool repeat = false;        // 是否去除重复字符
    int min_area = 64;           // 文字区域最小面积
    float text = 0.5;           // 文字检测阈值, 0<=text<=1
    float thresh = 0.5;	        // 文字区域识别阈值, 0<=thresh<=1
    float unclip_ratio = 2.0f;   // 区域扩展强度，1<=unclip_ratio
    const char* dictionary=nullptr;  	// 字典文件路径(dictionary.txt)
};
struct Polygon{
    float score;
    std::vector<cv::Point2f> points;
};
private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;
    MT::OCRDictionary dictionary;

    ParamsOCR params;
    std::vector<Polygon> polygons;
    std::vector<cv::Mat> input_images;
	std::vector<yo::Node> input_nodes_det,input_nodes_rec,input_nodes_cls;
	std::vector<yo::Node> output_nodes_det,output_nodes_rec,output_nodes_cls;

    Ort::Env env_det,env_cls,env_rec;
	Ort::Session* session_det = nullptr;
	Ort::Session* session_cls = nullptr;
	Ort::Session* session_rec = nullptr;

    Ort::SessionOptions options_det,options_cls,options_rec;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image) override;
    void postprocess(std::vector<Ort::Value>& output_tensors)override;

    std::vector<cv::Point2f> unclip(std::vector<cv::Point>& points);
    float box_score_slow(cv::Mat&pred,std::vector<cv::Point>& approx);
    std::vector<Polygon> postprocess_det(cv::Mat &pred,cv::Mat& bitmap);

    std::optional<std::vector<cv::Mat>> infer_det();                                // 文本区域识别,返回文本区域的分割
    std::optional<std::vector<cv::Mat>> infer_cls(std::vector<cv::Mat>& images);    // 文本方向识别
    std::optional<std::vector<Ort::Value>> infer_rec(std::vector<cv::Mat>& images); // 文本内容识别
public:
    PaddleOCR(){};
    ~PaddleOCR(){
        if(session_det != nullptr) delete session_det;
        if(session_cls != nullptr) delete session_cls;
        if(session_rec != nullptr) delete session_rec;
        session_det = nullptr;
        session_cls = nullptr;
        session_rec = nullptr;
    }
    PaddleOCR(const PaddleOCR&) = delete;// 删除拷贝构造函数
    PaddleOCR& operator=(const PaddleOCR&) = delete;// 删除赋值运算符
    int setparms(ParamsOCR parms);
    std::variant<bool,std::string> initialize(std::vector<std::string>& onnx_paths, bool is_cuda) override;
    std::variant<bool,std::string> inference(cv::Mat &image) override;
    
};