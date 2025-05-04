#pragma once
#include <print>
#include <memory>
#include <vector>
#include <iostream>
#include <functional>
#include <expected>
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "mtool.hpp"
#ifdef DEVICE_DML
#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"
#endif

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
namespace yo{

struct Node{
    char* name = nullptr;
    std::vector<int64_t> dim; // batch,channel,height,width
};

class Model{
public:
    virtual ~Model(){};
    virtual std::variant<bool,std::string> inference(cv::Mat &image) = 0;
    virtual std::variant<bool,std::string> initialize(std::vector<std::string>& onnx_paths,bool is_cuda) = 0;
protected:
    virtual void preprocess(cv::Mat &image)=0;
    virtual void postprocess(std::vector<Ort::Value>& output_tensors)=0;
    void load_onnx_info(Ort::Session* session,std::vector<Node>& input,std::vector<Node>& output,std::string onnx="default.onnx"){
        Ort::AllocatorWithDefaultOptions allocator;
        // 模型输入信息
        for (size_t index = 0; index < session->GetInputCount(); index++) {
            Ort::AllocatedStringPtr input_name_Ptr = session->GetInputNameAllocated(index, allocator);
            Ort::TypeInfo input_type_info = session->GetInputTypeInfo(index);
            auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
            Node node;
            node.dim = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
            const char* name = input_name_Ptr.get();
            size_t name_length = strlen(name) + 1;
            node.name = new char[name_length];
            strcpy(node.name, name);
            input.push_back(node);
        }
        // 模型输出信息
        for (size_t index = 0; index < session->GetOutputCount(); index++) {
            Ort::AllocatedStringPtr output_name_Ptr = session->GetOutputNameAllocated(index, allocator);
            Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(index);
            Node node;
            node.dim = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
            const char* name = output_name_Ptr.get();
            size_t name_length = strlen(name) + 1;
            node.name = new char[name_length];
            strcpy(node.name, name);
            output.push_back(node);
        }
        // 打印日志
        std::println("***************{}***************",onnx);
        for(auto &node:input){
            std::string dim_str = "[";
            for (size_t i = 0; i < node.dim.size(); ++i) {
                dim_str += std::to_string(node.dim[i]);
                if (i != node.dim.size() - 1) dim_str += ",";
            }
            dim_str += "]";
            std::println("input_name= [{}] ===> {}", node.name, dim_str);
        }
        for(auto &node:output){
            std::string dim_str = "[";
            for (size_t i = 0; i < node.dim.size(); ++i) {
                dim_str += std::to_string(node.dim[i]);
                if (i != node.dim.size() - 1) dim_str += ",";
            }
            dim_str += "]";
            std::println("output_name= [{}] ==> {}", node.name, dim_str);
        }
        std::println("************************************\n");
    }
};

// 没考虑线程安全的问题
// 环形队列
template <typename T, size_t N>
class FixedSizeQueue {
private:
    std::vector<T> data;
    size_t head;
    size_t tail;
    size_t count;
public:
    FixedSizeQueue():head(0), tail(0), count(0) {}
    bool push(T&& value) {
        if (this->full()){
            tail = (tail + 1) % N;
            data[head] = std::move(value);
        }else{
            if(data.size()<N){
                data.push_back(std::move(value));
                count++;
            }else{
                data[head] = std::move(value);
            }
        } 
        head = (head + 1) % N;
        return true;
    }
    bool push(const T& value) {
        return push(T(value));  // Create a copy and use the rvalue overload
    }
    T& at(size_t idx){
        if(idx >=this->count)
            throw std::out_of_range("Index out of range");
        idx = (tail + idx) % N;
        return data[idx];
    }
    bool empty() const {return count == 0;}
    bool full() const {return count == N;}
    size_t size() const {return count;}
};
}