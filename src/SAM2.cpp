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

int SAM2::initialize(std::string onnx_path, bool is_cuda)
{
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
    this->infer_status.current_frame++;
    // 图片预处理
    try {
        this->preprocess(image); 
     }catch (const std::exception& e) {
        std::println("Image preprocess failed!");
        return -2;
    }
    // 图片编码器，输入图片
    std::vector<Ort::Value> img_encoder_tensor;
    img_encoder_tensor.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        this->input_images[0].ptr<float>(),
                        this->input_images[0].total(),
                        this->img_encoder_input_nodes[0].dim.data(),
                        this->img_encoder_input_nodes[0].dim.size())
    );
    this->img_encoder_infer(img_encoder_tensor);
    this->mem_attention_infer();
    this->img_decoder_infer();
    this->mem_encoder_infer();
    // 后处理
    std::vector<Ort::Value> output_tensors(1);
    output_tensors[0] = std::move(this->img_decoder_out[2]);
    try {
        this->postprocess(output_tensors);
    }catch (const std::exception& e) {
        std::println("tensor postprocess failed!!");
        return -4;
    }
    return 0;
}
std::vector<Ort::Value> SAM2::build_mem_attention_input(){
    // [num_obj_ptr, current_vision_feat, current_vision_pos_embed, memory, memory_pos_embed]->[image_embed]
    std::vector<Ort::Value> input_tensor(5);
    input_tensor[1] = std::move(this->img_encoder_out[3]); //current_vision_feat
    input_tensor[2] = std::move(this->img_encoder_out[4]); //current_vision_pos_embed
    if(this->infer_status.current_frame>1){
        //num_obj_ptr,缓存帧的数量，最小应该是2
        input_tensor[0] = Ort::Value::CreateTensor<int32_t>(
                            memory_info,
                            &infer_status.current_frame,
                            1,
                            this->mem_attention_input_nodes[0].dim.data(),
                            this->mem_attention_input_nodes[0].dim.size());
        // input_tensor[3] = todo..
        // input_tensor[4] = todo..
    }
    return input_tensor;
}
// input [1,3,1024,1024]
// output: 
//      pix_feat        [1,256,64,64]
//      high_res_feat0  [1,32,256,256]
//      high_res_feat1  [1,64,128,128]
//      vision_feats    [4096,1,256]
//      vision_pos_embed [4096,1,256]
void SAM2::img_encoder_infer(std::vector<Ort::Value> &input_tensor){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->img_encoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->img_encoder_output_nodes) output_names.push_back(node.name);
    try {
        this->img_encoder_out = this->img_encoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(), 
            output_names.data(), 
            output_names.size()); 
    }catch (const std::exception& e) {
        std::println("ERROR: img_encoder_infer failed!!");
    }
}

// input:
//      num_obj_ptr             [num]
//      current_vision_feat     [4096,1,256]
//      current_vision_pos_embed [4096,1,256]
//      memory                  [buff,1,64]
//      memory_pos_embed        [buff,1,64]
// output:
//      image_embed     [1,256,64,64]
void SAM2::mem_attention_infer(){
    //创建输入数据
    auto input_tensor = build_mem_attention_input();
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->mem_attention_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->mem_attention_output_nodes) output_names.push_back(node.name);
    try {
        this->mem_attention_out = this->mem_attention_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size());
    }catch (const std::exception& e) {
        std::println("ERROR: mem_attention_infer failed!!");
    }
}

// input:
//      point_coords    [num_labels,num_points,2]
//      point_labels    [num_labels,num_points]
//      frame_size      [2]
//      image_embed     [1,256,64,64]
//      high_res_feats_0[1,32,256,256]
//      high_res_feats_1[1,64,128,128]
// output:
//      obj_ptr       [1,256]
//      mask_for_mem  [1,1,1024,1024]
//      pred_mask     [1,H,W]
void SAM2::img_decoder_infer(){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->img_decoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->img_decoder_output_nodes) output_names.push_back(node.name);

    std::vector<Ort::Value> input_tensor(6);

    auto& box = parms.prompt_box;
    std::vector<float>point_val{(float)box.x,(float)box.y,(float)box.x+box.width,(float)box.y+box.height};//xyxy
    std::vector<float> point_labels = {2,3};
    std::vector<int32_t> frame_size = {ori_img->rows,ori_img->cols};
    
    input_tensor[0] = Ort::Value::CreateTensor<float>(memory_info,point_val.data(),point_val.size(),
                        this->img_decoder_input_nodes[0].dim.data(),
                        this->img_decoder_input_nodes[0].dim.size());
    input_tensor[1] = Ort::Value::CreateTensor<float>(memory_info,point_labels.data(),point_labels.size(),
                        this->img_decoder_input_nodes[1].dim.data(),
                        this->img_decoder_input_nodes[1].dim.size());
    input_tensor[2] = Ort::Value::CreateTensor<int32_t>(memory_info,frame_size.data(),frame_size.size(),
                        this->img_decoder_input_nodes[2].dim.data(),
                        this->img_decoder_input_nodes[2].dim.size());
    input_tensor[3] = std::move(this->mem_attention_out[0]);
    input_tensor[4] = std::move(this->img_encoder_out[1]);
    input_tensor[5] = std::move(this->img_encoder_out[2]);
    //***************************************************
    try {
        this->img_decoder_out = this->img_encoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size());
    }catch (const std::exception& e) {
        std::println("ERROR: img_decoder_infer failed!!");
    }
}


// input:
//      mask_for_mem    [1,1,1024,1024]
//      pix_feat        [1,256,64,64]
// output:
//      maskmem_features  [4096,1,64]
//      maskmem_pos_enc   [4096,1,64]
void SAM2::mem_encoder_infer(){
    std::vector<Ort::Value> input_tensor(2);
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->mem_encoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->mem_encoder_output_nodes) output_names.push_back(node.name);
    input_tensor[0] = std::move(this->img_decoder_out[1]);
    input_tensor[1] = std::move(this->img_encoder_out[0]);
    //***************************************************
    try {
        this->mem_encoder_out = this->mem_encoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size());
    }catch (const std::exception& e) {
        std::println("ERROR: mem_encoder_infer failed!!");
    }
}

void SAM2::preprocess(cv::Mat &image){
    input_images.clear();
    std::vector<cv::Mat> mats{image};
    cv::Mat img = cv::dnn::blobFromImages(mats, 1/255.0,cv::Size(1024,1024), cv::Scalar(0, 0, 0), true, false);
    input_images.push_back(std::move(img));
}

void SAM2::postprocess(std::vector<Ort::Value> &output_tensors){

    std::println("postprocess");

}
