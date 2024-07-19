#pragma once
#include "Model.h"
#include <fstream>
#include <print>

const static std::map<int,std::string>  LABEL = {
        {0,"person"},{1,"bicycle"},{2,"car"},{3,"moto"},{4,"airplane"},{5,"bus"},{6,"train"},{7,"truck"},
        {8,"boat"},{9,"traffic light"},{10,"fire hydrant"},{11,"stop sign"},{12,"parking meter"},{13,"bench"},{14,"bird"},{15,"cat"},
        {16,"dog"},{17,"horse"},{18,"sheep"},{19,"cow"},{20,"elephant"},{21,"bear"},{22,"zebra"},{23,"giraffe"},
        {24,"backpack"},{25,"umbrella"},{26,"handbag"},{27,"tie"},{28,"suitcase"},{29,"frisbee"},{30,"skis"},{31,"snowboard"},
        {32,"sports ball"},{33,"kite"},{34,"baseball bat"},{35,"baseball glove"},{36,"skateboard"},{37,"surfboard"},{38,"tennis racket"},{39,"bottle"},
        {40,"wine glass"},{41,"cup"},{42,"fork"},{43,"knife"},{44,"spoon"},{45,"bowl"},{46,"banana"},{47,"apple"},
        {48,"sandwich"},{49,"orange"},{50,"broccoli"},{51,"carrot"},{52,"hot dog"},{53,"pizza"},{54,"donut"},{55,"cake"},
        {56,"chair"},{57,"couch"},{58,"potted plant"},{59,"bed"},{60,"dining table"},{61,"toilet"},{62,"tv"},{63,"laptop"},
        {64,"mouse"},{65,"remote"},{66,"keyboard"},{67,"phone"},{68,"microwave"},{69,"oven"},{70,"toaster"},{71,"sink"},
        {72,"refrigerator"},{73,"book"},{74,"clock"},{75,"vase"},{76,"scissors"},{77,"teddy bear"},{78,"hair drier"},{79,"toothbrush"}
    };


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