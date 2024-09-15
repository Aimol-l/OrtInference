#include <iostream>
#include <filesystem>
#include "SAM2.h"
#include "Yolov10.h"
#include "Yolov10SAM.h"
#include "Yolov10Trace.h"

void yolo(){
    auto yolov10 = std::make_unique<Yolov10>();
    std::vector<std::string> onnx_paths{"../models/yolov10/yolov10m.onnx"};
    auto r = yolov10->initialize(onnx_paths,true);
    if(r.index() != 0){
        std::string error = std::get<std::string>(r);
        std::println("错误：{}",error);
        return;
    }
    yolov10->setparms({.score=0.5f,.nms=0.8f});
    
    std::string folder_path = "../assets/input/*.jpg";
    std::string output_path = "../assets/output/";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto& path : paths) {
        std::println("path={}",path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = yolov10->inference(image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("duration = {}ms",duration);
        if(result.index() == 0){
            auto filename = std::filesystem::path(path).filename().string();
            cv::imwrite(output_path+filename,image);
            cv::imshow("Image", image);
            cv::waitKey(0);
        }else{
            std::string error = std::get<std::string>(result);
            std::println("错误：{}",error);
            continue;
        }
    }
}
void yolosam(){
    auto yolov10sam = std::make_unique<Yolov10SAM>();
    std::vector<std::string> onnx_paths{
        "../models/yolov10/yolov10m.onnx",
        "../models/sam/ESAM_encoder.onnx",
        "../models/sam/ESAM_deocder.onnx"
    };
    auto r = yolov10sam->initialize(onnx_paths,true);
    if(r.index() != 0){
        std::string error = std::get<std::string>(r);
        std::println("错误：{}",error);
        return;
    }
    yolov10sam->setparms({.score=0.5f,.nms=0.8f});
    
    std::string folder_path = "../assets/input/*.jpg";
    std::string output_path = "../assets/output/";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto& path : paths) {
        std::println("path={}",path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = yolov10sam->inference(image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("duration = {}ms",duration);
        if(result.index() == 0){
            // auto filename = std::filesystem::path(path).filename().string();
            // cv::imwrite(output_path+filename,image);
            // cv::imshow("Image", image);
            // cv::waitKey(0);
        }else{
            std::string error = std::get<std::string>(result);
            std::println("错误：{}",error);
            continue;
        }
    }
}
void yolotrace(){
    auto yolov10trace = std::make_unique<Yolov10Trace>();
    std::vector<std::string> onnx_paths{"../models/yolov10/yolov10m.onnx"};
    auto r = yolov10trace->initialize(onnx_paths,true);
    if(r.index() != 0){
        std::string error = std::get<std::string>(r);
        std::println("错误：{}",error);
        return;
    }
    std::string video_path = "../assets/video/test.mp4";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) return;
    //************************************************************
    std::cout << "视频中图像的宽度=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频中图像的高度=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率=" << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "视频的总帧数=" << capture.get(cv::CAP_PROP_FRAME_COUNT)<<std::endl;
    //************************************************************
    yolov10trace->setparms({
        .camera_fps = 25,
        .buffer_size = 20,
        .score=0.5f,
        .nms=0.5f});
    //************************************************************
    cv::Mat frame;
    while (true) {
        if (!capture.read(frame) || frame.empty()) break;
        auto result = yolov10trace->inference(frame);
        if(result.index() == 0){
            cv::imshow("frame", frame);
            int key = cv::waitKey(10);
            if (key == 'q' || key == 27) break;
        }else{
            std::string error = std::get<std::string>(result);
            std::println("错误：{}",error);
            break;
        }
    }
    capture.release();
}   
void sam2(){
    auto sam2 = std::make_unique<SAM2>();
    std::vector<std::string> onnx_paths{
        "../models/sam2/small/image_encoder.onnx",
        "../models/sam2/small/memory_attention.onnx",
        "../models/sam2/small/image_decoder.onnx",
        "../models/sam2/small/memory_encoder.onnx"
    };
    auto r = sam2->initialize(onnx_paths,true);
    if(r.index() != 0){
        std::string error = std::get<std::string>(r);
        std::println("错误：{}",error);
        return;
    }
    sam2->setparms({.type=1,
                    .prompt_box = {745,695,145,230},
                    .prompt_point = {846,794}}); // 在原始图像上的box,point

    std::string video_path = "../assets/video/test.mkv";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) return;
    //************************************************************
    std::cout << "视频中图像的宽度=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频中图像的高度=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率=" << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "视频的总帧数=" << capture.get(cv::CAP_PROP_FRAME_COUNT)<<std::endl;
    //************************************************************
    cv::Mat frame;
    size_t idx = 0;
    while (true) {
        if (!capture.read(frame) || frame.empty()) break;
        auto start = std::chrono::high_resolution_clock::now();
        auto result = sam2->inference(frame);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("frame = {},duration = {}ms",idx++,duration);
        if(result.index() == 0){
            std::string text = std::format("frame = {},fps={:.1f}",idx,1000.0f/duration);
            cv::putText(frame,text,cv::Point{30,40},1,2,cv::Scalar(0,0,255),2);
            cv::imshow("frame", frame);
            int key = cv::waitKey(5);
            if (key == 'q' || key == 27) break;
        }else{
            std::string error = std::get<std::string>(result);
            std::println("错误：{}",error);
            break;
        }
    }
    capture.release();
}
int main(int argc, char const *argv[]){
    // yolo();
    // yolosam();
    // yolotrace();
    sam2();
    return 0;   
}

