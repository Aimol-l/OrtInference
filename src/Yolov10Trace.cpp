#include "Yolov10Trace.h"

int Yolov10Trace::setparms(ParamsTrace parms){
    tracker = std::make_unique<BYTETracker>(parms.camera_fps,parms.buffer_size);
    this->parms = std::move(parms);
    return 1;
}

std::variant<bool,std::string> Yolov10Trace::initialize(std::vector<std::string>& onnx_paths, bool is_cuda){
    auto is_file = [](const std::string& filename) {
        std::ifstream file(filename.c_str());
        return file.good();
    };
    std::string& onnx_path = onnx_paths[0];  
    if (!is_file(onnx_path)) {
        return std::format("Model file dose not exist.file:{}",onnx_path);
    }
    this->session_options.SetIntraOpNumThreads(4);
    if (is_cuda) {
        try {
            OrtCUDAProviderOptions options;
            options.device_id = 0;
            options.arena_extend_strategy = 0;
            options.gpu_mem_limit = SIZE_MAX;
            options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            options.do_copy_in_default_stream = 1;
            session_options.AppendExecutionProvider_CUDA(options);
            std::println("Using CUDA...");
        }catch (const std::exception& e) {
            std::string error(e.what());
            return error;
        }
    }else {
        std::println("Using CPU...");
    }
    try {
#ifdef _WIN32
        unsigned len = onnx_path.size() * 2; // 预留字节数
        setlocale(LC_CTYPE, ""); //必须调用此函数,本地化
        wchar_t* p = new wchar_t[len]; // 申请一段内存存放转换后的字符串
        mbstowcs(p, onnx_path.c_str(), len); // 转换
        std::wstring wstr(p);
        delete[] p; // 释放申请的内存
        session = new Ort::Session(env, wstr.c_str(), this->session_options);
#else
        session = new Ort::Session(env, (const char*)onnx_path.c_str(), this->session_options);
#endif
    }catch (const std::exception& e) {
        return std::format("Failed to load model. Please check your onnx file!");
    }
    //************************************************
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = this->session->GetInputCount();
    for (size_t index = 0; index < num_input_nodes; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->session->GetInputNameAllocated(index, allocator);
        // 复制名称到新的缓冲区
        Ort::TypeInfo input_type_info = this->session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        //*****************
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        //*************************
        std::print("{} = ",node.name);
        for(size_t j=0;j<input_dims.size()-1;j++) std::print("{}x",node.dim[j]);
        std::println("{}",node.dim[input_dims.size()-1]);
        this->input_nodes.push_back(node);
    }
    //************************************************
    size_t num_output_nodes = this->session->GetOutputCount();
    for (size_t index = 0; index < num_output_nodes; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->session->GetOutputNameAllocated(index, allocator);
        // 复制名称到新的缓冲区
        Ort::TypeInfo output_type_info = this->session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        //*****************
        yo::Node node;
        for(size_t j=0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        //*************************
        std::print("{} = ",node.name);
        for(size_t j=0;j<output_dims.size()-1;j++) std::print("{}x",node.dim[j]);
        std::println("{}",node.dim[output_dims.size()-1]);
        this->output_nodes.push_back(node);
    }
    //************************************************
    this->is_inited = true;
    std::println("initialize ok!!");
    return true;
}

std::variant<bool,std::string> Yolov10Trace::inference(cv::Mat &image){
    if (image.empty() || !is_inited) return "image can not empyt!";
    this->ori_img = &image;

    // 图片预处理
    // 1. 直接resize
    // 2. 图像padding
    try {
        this->preprocess(image); 
     }catch (const std::exception& e) {
        return "Image preprocess failed!";
    }

    // 创建模型输入张量
    // cv::Mat --> Ort::Value
    std::vector<Ort::Value> input_tensor;
    input_tensor.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        this->input_images[0].ptr<float>(),
        this->input_images[0].total(),
        this->input_nodes[0].dim.data(),
        this->input_nodes[0].dim.size())
    );

    // 推理
    std::vector<const char*> input_names,output_names;
    for(int i = 0;i<this->input_nodes.size();i++)
        input_names.push_back(this->input_nodes[i].name);
    for(int i = 0;i<this->input_nodes.size();i++)
        output_names.push_back(this->output_nodes[i].name);
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = this->session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),  // images
            input_tensor.data(), // 1*3*640*640
            input_tensor.size(), // 1
            output_names.data(), // output0
            output_names.size()); // 1
    }catch (const std::exception& e) {
        return "forward model failed!!";
    }
    // 输出后处理
    try {
        this->postprocess(output_tensors);
    }catch (const std::exception& e) {
        return "tensor postprocess failed!!";
    }
    return true;
}

void Yolov10Trace::preprocess(cv::Mat &image){
    cv::Mat image_ = image.clone();
    auto net_w = (float)this->input_nodes[0].dim.at(3); // 640
    auto net_h = (float)this->input_nodes[0].dim.at(2); // 640
    float scale = std::min(net_w/image.cols,net_h/image.rows);
    cv::resize(image_,image_,cv::Size(int(image.cols*scale),int(image.rows*scale)));
    int top     = (net_h - image_.rows) / 2;
    int bottom  = (net_h - image_.rows) / 2 + int(net_h - image_.rows) % 2;
    int left    = (net_w - image_.cols) / 2;
    int right   = (net_w - image_.cols) / 2 + int(net_w - image_.cols) % 2; 
    cv::copyMakeBorder(image_, image_, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114)); // padding
    input_images.clear();
    std::vector<cv::Mat> mats{image_};
    cv::Mat blob = cv::dnn::blobFromImages(mats, 1/255.0,cv::Size(net_w, net_h), cv::Scalar(0, 0, 0), true, false);//1*3*640*640
    input_images.push_back(std::move(blob));
}

void Yolov10Trace::postprocess(std::vector<Ort::Value> &output_tensors){
    float* output = output_tensors[0].GetTensorMutableData<float>(); // 1*300*6
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<cv::Rect> boxes;
    auto net_w = (float)this->input_nodes[0].dim.at(3); // 1024
    auto net_h = (float)this->input_nodes[0].dim.at(2); // 1024
    float scale = std::min(net_w/ori_img->cols,net_h/ori_img->rows);

    for(size_t index = 0;index < this->output_nodes[0].dim[1]; index +=this->output_nodes[0].dim[2]){
        auto x1 = output[index];
        auto y1 = output[index + 1] ;
        auto x2  = output[index + 2];
        auto y2  = output[index + 3];
        auto score = output[index + 4];
        auto label = (int)output[index + 5];

        auto pad_w = (net_w - ori_img->cols * scale) / 2;
        auto pad_h = (net_h - ori_img->rows * scale) / 2;
        //从预处理图片还原到原始图片上
        x1 = (x1 - pad_w) / scale;
        x2 = (x2 - pad_w) / scale;
        y1 = (y1 - pad_h) / scale;
        y2 = (y2 - pad_h) / scale;

        cv::Rect rect(cv::Point2f(x1,y1),cv::Size2f(x2-x1,y2-y1));
        boxes.emplace_back(rect);
        labels.emplace_back(label);
        scores.emplace_back(score);
    }
    //*****************************************************
    std::vector<int>indices;
    cv::dnn::NMSBoxes(boxes,scores,this->parms.score,this->parms.nms,indices);
    //*****************************************************
    std::vector<Object> objects;
    for(auto i:indices){
        objects.emplace_back(Object{boxes[i],labels[i],scores[i]});
    }
    std::vector<STrack> output_stracks = tracker->update(objects);
    // draw boxes
    for(size_t i =0;i<output_stracks.size();i++){
        std::string name = LABEL.at(output_stracks[i].label_id);
        std::size_t hash = std::hash<std::string>{}(name);
        double g = (hash & 0xFF0000) >> 16;
        double r = (hash & 0x00FF00) >> 8;
        double b = hash & 0x0000FF;
        ///**************************************************
        std::vector<float> box = output_stracks[i].tlwh;
        int x = (int)box[0],y=(int)box[1],w=(int)box[2],h=(int)box[3];
        cv::rectangle(*ori_img,cv::Rect{x,y-20,w,20},cv::Scalar{25,255,188},-1);
        cv::rectangle(*ori_img,cv::Rect{x,y,w,h},cv::Scalar{b,g,r},2);
        cv::putText(*ori_img,std::format("{}{}:{:.2f}",output_stracks[i].track_id,
            name,output_stracks[i].score),
            cv::Point{x,y-5},1,1.2,cv::Scalar{b,g,r});
    }
}
