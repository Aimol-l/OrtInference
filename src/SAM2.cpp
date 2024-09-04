#include "SAM2.h"

std::vector<std::string> SAM2::str_split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}
int SAM2::initialize(std::string onnx_path, bool is_cuda){
    // 
    std::vector<std::string> onnx_paths = this->str_split(onnx_path,'|');
    assert(onnx_paths.size() == 4);
    auto is_file = [](const std::string& filename) {
        std::ifstream file(filename.c_str());
        return file.good();
    };
    for (const auto& path : onnx_paths) {
        if (!is_file(path)) {
            std::println("Model file dose not exist.file:{}",path);
            return -2;
        }
    }
    this->img_encoder_options.SetIntraOpNumThreads(2); //设置线程数量
    this->img_decoder_options.SetIntraOpNumThreads(2); //设置线程数量
    this->mem_attention_options.SetIntraOpNumThreads(2); //设置线程数量
    this->mem_encoder_options.SetIntraOpNumThreads(2); //设置线程数量
    //***********************************************************
    if(is_cuda) {
        try {
            OrtCUDAProviderOptions options;
            options.device_id = 0;
            options.arena_extend_strategy = 0;
            options.gpu_mem_limit = SIZE_MAX;
            options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            options.do_copy_in_default_stream = 1;
            this->img_encoder_options.AppendExecutionProvider_CUDA(options);
            this->img_decoder_options.AppendExecutionProvider_CUDA(options);
            this->mem_attention_options.AppendExecutionProvider_CUDA(options);
            this->mem_encoder_options.AppendExecutionProvider_CUDA(options);
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
    // todo
    return -4;
#else
        img_encoder_session = new Ort::Session(img_encoder_env, (const char*)onnx_paths[0].c_str(), this->img_encoder_options);
        img_decoder_session = new Ort::Session(img_decoder_env, (const char*)onnx_paths[1].c_str(), this->img_decoder_options);
        mem_attention_session = new Ort::Session(mem_attention_env, (const char*)onnx_paths[2].c_str(), this->mem_attention_options);
        mem_encoder_session = new Ort::Session(mem_encoder_env, (const char*)onnx_paths[3].c_str(), this->mem_encoder_options);
#endif  
    }catch (const std::exception& e) {
        std::println("Failed to load model. Please check your onnx file!");
        return -4;
    }
    //************************************************************
    Ort::AllocatorWithDefaultOptions allocator;
    size_t const img_encoder_input_num = this->img_encoder_session->GetInputCount();
    size_t const img_decoder_input_num = this->img_decoder_session->GetInputCount();
    size_t const mem_attention_input_num = this->mem_attention_session->GetInputCount();
    size_t const mem_encoder_input_num = this->mem_encoder_session->GetInputCount();
    //************************************************************
    for (size_t index = 0; index < img_encoder_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->img_encoder_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->img_encoder_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->img_encoder_input_nodes.push_back(node);
    }
    for (size_t index = 0; index < img_decoder_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->img_decoder_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->img_decoder_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->img_decoder_input_nodes.push_back(node);
    }
    for (size_t index = 0; index < mem_attention_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->mem_attention_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->mem_attention_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->mem_attention_input_nodes.push_back(node);
    }
    for (size_t index = 0; index < mem_encoder_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->mem_encoder_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->mem_encoder_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->mem_encoder_input_nodes.push_back(node);
    }
    //************************************************************
    size_t const img_encoder_output_num = this->img_encoder_session->GetOutputCount();
    size_t const img_decoder_output_num = this->img_decoder_session->GetOutputCount();
    size_t const mem_attention_output_num = this->mem_attention_session->GetOutputCount();
    size_t const mem_encoder_output_num = this->mem_encoder_session->GetOutputCount();
    //************************************************************
    for (size_t index = 0; index < img_encoder_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->img_encoder_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->img_encoder_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j = 0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->img_encoder_output_nodes.push_back(node);
    }
    for (size_t index = 0; index < img_decoder_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->img_decoder_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->img_decoder_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j = 0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->img_decoder_output_nodes.push_back(node);
    }
    for (size_t index = 0; index < mem_attention_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->mem_attention_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->mem_attention_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j = 0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->mem_attention_output_nodes.push_back(node);
    }
    for (size_t index = 0; index < mem_encoder_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->mem_encoder_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->mem_encoder_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j = 0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->mem_encoder_output_nodes.push_back(node);
    }
    
    //************************************************************
    for(const auto& outputs: img_encoder_input_nodes){
        std::print("{}=",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++)  std::print("[{},",outputs.dim[i]);
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    for(const auto& outputs: img_decoder_input_nodes){
        std::print("{}=",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++)  std::print("[{},",outputs.dim[i]);
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    for(const auto& outputs: mem_attention_input_nodes){
        std::print("{}=",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++)  std::print("[{},",outputs.dim[i]);
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    for(const auto& outputs: mem_encoder_input_nodes){
        std::print("{}=",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++)  std::print("[{},",outputs.dim[i]);
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    this->is_inited = true;
    std::println("initialize ok!!");
    return 1;
}

int SAM2::inference(cv::Mat &image){
    if (image.empty() || !is_inited) return -1;
    this->ori_img = &image;
    // 图片预处理
    try {
        this->preprocess(image); 
     }catch (const std::exception& e) {
        std::println("Image preprocess failed!");
        return -2;
    }
    // 图片编码器
    
    // mem_attention

    // 图片解码器

    // mem_encoder

    // 后处理

    return 0;
}

void SAM2::preprocess(cv::Mat &image){
    input_images.clear();
    std::vector<cv::Mat> mats{image};
    cv::Mat img = cv::dnn::blobFromImages(mats, 1/255.0,cv::Size(1024,1024), cv::Scalar(0, 0, 0), true, false);
    input_images.push_back(std::move(img));
}

void SAM2::postprocess(std::vector<Ort::Value> &output_tensors){

}
