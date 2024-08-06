#include "GroundDino.h"

int GroundDino::setparms(Params_dino parms){
    this->parms = std::move(parms);
    return 1;
}

int GroundDino::initialize(std::string onnx_path, bool is_cuda){
    
    auto is_file = [](const std::string& filename) {
        std::ifstream file(filename.c_str());
        return file.good();
    };
    if (!is_file(onnx_path)) {
        std::println("Model file dose not exist.file:{}",onnx_path);
        return -2;
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
            std::cerr << e.what() << '\n';
            return -3;
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
        std::println("Failed to load model. Please check your onnx file!");
        return -4;
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
        this->output_nodes.push_back(node);
    }
    //****************************打印模型信息*******************************
    std::println("*************************dino*************************");
    for(const auto& inputs: input_nodes){
        std::print("{}= [",inputs.name);
        for(size_t i=0;i<inputs.dim.size()-1;i++){
            std::print("{},",inputs.dim[i]);
        }
        std::println("{}]",inputs.dim[inputs.dim.size()-1]);
    }
    std::println("-----------------------------------------------------");
    for(const auto& outputs: output_nodes){
        std::print("{}= [",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++){
            std::print("{},",outputs.dim[i]);
        }
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    std::println("******************************************************");
    //************************************************
    this->is_inited = true;
    std::println("initialize ok!!");
    return 1;
}

int GroundDino::inference(cv::Mat &image){
    if (image.empty() || !is_inited) return -1;
    this->ori_img = &image;

    // 图片预处理
    // 1. 直接resize
    // 2. 图像padding
    try {
        this->preprocess(image); 
        std::println("Image preprocess okk!");
     }catch (const std::exception& e) {
        std::println("Image preprocess failed!");
        return -2;
    }

    // 运行时确定模型输入张量和大小
    std::vector<Ort::Value> input_tensor;
    int64_t seq_len = COCO.contains(parms.prompt)? COCO.at(parms.prompt).size():COCO.at("biscuits").size();
    input_nodes[0].dim = {1,3,image.rows,image.cols};   // img
    input_nodes[1].dim = {1,seq_len};                   // input_ids
    input_nodes[2].dim = {1,seq_len};                   // attention_mask
    input_nodes[3].dim = {1,seq_len};                   // position_ids
    input_nodes[4].dim = {1,seq_len};                   // token_type_ids
    input_nodes[5].dim = {1,seq_len,seq_len};           // text_token_mask
    std::println("seq_len = {}",seq_len);
    {
        // img
        input_tensor.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            this->input_images[0].ptr<float>(),
            this->input_images[0].total(),
            this->input_nodes[0].dim.data(),
            this->input_nodes[0].dim.size())
        );
        std::println("img tensor okkk");
        // input_ids
        auto input_ids = COCO.at(parms.prompt);
        input_tensor.push_back(Ort::Value::CreateTensor<int64>(
            memory_info,
            input_ids.data(),
            input_ids.size(),
            this->input_nodes[1].dim.data(),
            this->input_nodes[1].dim.size())
        );
        std::println("input_ids tensor okkk");
        // attention_mask
        auto attention_mask = std::vector<int64_t>(seq_len,1); // 不能使用 bool类型，vector特殊优化导致
        input_tensor.push_back(Ort::Value::CreateTensor(
            memory_info,
            attention_mask.data(),
            attention_mask.size(),
            this->input_nodes[2].dim.data(),
            this->input_nodes[2].dim.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
        );
        std::println("attention_mask tensor okkk");
        // position_ids，token之间的注意力关系
        auto position_ids = std::vector<int64_t>(seq_len,0);
        for(size_t i = 2;i < position_ids.size() - 1;i++){
            position_ids[i] = i-1;
        }
        input_tensor.push_back(Ort::Value::CreateTensor<int64>(
            memory_info,
            position_ids.data(),
            position_ids.size(),
            this->input_nodes[3].dim.data(),
            this->input_nodes[3].dim.size())
        );
        std::println("position_ids tensor okkk");
        // token_type_ids
        auto token_type_ids = std::vector<int64_t>(seq_len,0);
        input_tensor.push_back(Ort::Value::CreateTensor<int64>(
            memory_info,
            token_type_ids.data(),
            token_type_ids.size(),
            this->input_nodes[4].dim.data(),
            this->input_nodes[4].dim.size())
        );
        std::println("token_type_ids tensor okkk");
        // text_token_mask ，主对角是1,第二圈内开始全部是1
        auto text_token_mask = std::vector<int64_t>(seq_len*seq_len,0);
        for(size_t i =0;i<seq_len;i++){
            for(size_t j =0;j<seq_len;j++){
                if(i==j) text_token_mask[i*seq_len+j] = 1;
                if(i >0 && i<seq_len-1 && j>0&&j<seq_len-1) text_token_mask[i*seq_len+j] = 1;
            }
        }
        input_tensor.push_back(Ort::Value::CreateTensor(
            memory_info,
            text_token_mask.data(),
            text_token_mask.size(),
            this->input_nodes[5].dim.data(),
            this->input_nodes[5].dim.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
        );
        std::println("text_token_mask tensor okkk");
    }
    // 推理**************************
    std::vector<const char*> input_names,output_names;
    for(int i = 0;i<this->input_nodes.size();i++) input_names.push_back(this->input_nodes[i].name);
    for(int i = 0;i<this->output_nodes.size();i++) output_names.push_back(this->output_nodes[i].name);
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = this->session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size());
    }catch (const std::exception& e) {
        std::println("{}",e.what());
        std::println("forward model failed!!");
        return -3;
    }
    // 输出后处理
    try {
        this->postprocess(output_tensors);
    }catch (const std::exception& e) {
        std::println("tensor postprocess failed!!");
        return -4;
    }
    return 1;
}

void GroundDino::preprocess(cv::Mat &image){
    cv::resize(image,image,cv::Size(1200,800));
    std::vector<cv::Mat> mats{image};
    cv::Mat blob = cv::dnn::blobFromImages(mats, 1/255.0,image.size(), cv::Scalar(0, 0, 0), true, false);//1*3*640*640
    input_images.push_back(std::move(blob));
}

void GroundDino::postprocess(std::vector<Ort::Value> &output_tensors){
    float* out_logits = output_tensors[0].GetTensorMutableData<float>(); // [1,900,256]
    float* out_boxes = output_tensors[1].GetTensorMutableData<float>(); // [1,900,4]
    std::vector<int64_t> logits_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> boxes_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    // 获取出bboxes
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    for(size_t i = 0;i < boxes_shape[1]; i++){
        auto index = i * boxes_shape[2];
        auto x = out_boxes[index] ;
        auto y = out_boxes[index + 1] ;
        auto w  = out_boxes[index + 2] ;
        auto h  = out_boxes[index + 3] ;
        
        // 计算 score
        float* log_begin = out_logits + i * logits_shape[2];
        float* log_end   = out_logits + (i+1) * logits_shape[2];
        float max_score = *std::max_element(log_begin,log_end);
        std::println("[{},{},{},{}],p={}",x,y,w,h,max_score);
        if(max_score < this->parms.score) continue;
        scores.push_back(max_score);
        bboxes.push_back({x,y,w,h});
    }
    // nms
    std::vector<int>indices;
    cv::dnn::NMSBoxes(bboxes,scores,this->parms.score,this->parms.nms,indices);
    // draw
    for(const auto i:indices){
        ///**************************************************
        cv::rectangle(*ori_img,bboxes[i],cv::Scalar{0,0,255},2);
        cv::putText(*ori_img,std::format("P={:.2f}",scores[i]),cv::Point{bboxes[i].x,bboxes[i].y-5}
            ,1,1.1,cv::Scalar{0,255,0});
    }
}
