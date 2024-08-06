#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

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
const static std::map<std::string,std::vector<int64_t>> COCO = 
        {{"person",{101,2711,102}},{"bicycle",{101,10165,102}},{"car",{101,2482,102}},{"moto",{101,9587,3406,102}},
        {"airplane",{101,13297,102}},{"bus",{101,3902,102}},{"train",{101,3345,102}},{"truck",{101,4744,102}},
        {"boat",{101,4049,102}},{"traffic light",{101,4026,2422,102}},{"fire hydrant",{101,2543,26018,3372,102}},
        {"stop sign",{101,2644,3696,102}},{"parking meter",{101,5581,8316,102}},{"bench",{101,6847,102}},
        {"bird",{101,4743,102}},{"cat",{101,4937,102}},{"dog",{101,3899,102}},{"horse",{101,3586,102}},
        {"sheep",{101,8351,102}},{"cow",{101,11190,102}},{"elephant",{101,10777,102}},{"bear",{101,4562,102}},
        {"zebra",{101,29145,102}},{"giraffe",{101,21025,27528,7959,102}},{"backpack",{101,13383,102}},
        {"umbrella",{101,12977,102}},{"handbag",{101,2192,16078,102}},{"tie",{101,5495,102}},{"suitcase",{101,15940,102}},
        {"frisbee",{101,10424,2483,11306,102}},{"skis",{101,8301,2015,102}},{"snowboard",{101,4586,6277,102}},
        {"sports ball",{101,2998,3608,102}},{"kite",{101,20497,102}},{"baseball bat",{101,3598,7151,102}},
        {"baseball glove",{101,3598,15913,102}},{"skateboard",{101,17260,6277,102}},{"surfboard",{101,14175,6277,102}},
        {"tennis racket",{101,5093,14513,3388,102}},{"bottle",{101,5835,102}},{"wine glass",{101,4511,3221,102}},
        {"cup",{101,2452,102}},{"fork",{101,9292,102}},{"knife",{101,5442,102}},{"spoon",{101,15642,102}},
        {"bowl",{101,4605,102}},{"banana",{101,15212,102}},{"apple",{101,6207,102}},{"sandwich",{101,11642,102}},
        {"orange",{101,4589,102}},{"broccoli",{101,22953,21408,3669,102}},{"carrot",{101,25659,102}},
        {"hot dog",{101,2980,3899,102}},{"pizza",{101,10733,102}},{"donut",{101,2123,4904,102}},{"cake",{101,9850,102}},
        {"chair",{101,3242,102}},{"couch",{101,6411,102}},{"potted plant",{101,8962,3064,3269,102}},
        {"bed",{101,2793,102}},{"dining table",{101,7759,2795,102}},{"toilet",{101,11848,102}},{"tv",{101,2694,102}},
        {"laptop",{101,12191,102}},{"mouse",{101,8000,102}},{"remote",{101,6556,102}},{"keyboard",{101,9019,102}},
        {"phone",{101,3042,102}},{"microwave",{101,18302,102}},{"oven",{101,17428,102}},{"toaster",{101,15174,2121,102}},
        {"sink",{101,7752,102}},{"refrigerator",{101,18097,102}},{"book",{101,2338,102}},{"clock",{101,5119,102}},
        {"vase",{101,18781,102}},{"scissors",{101,25806,102}},{"teddy bear",{101,11389,4562,102}},
        {"hair drier",{101,2606,2852,3771,102}},{"toothbrush",{101,11868,18623,102}},

        {"areca",{101,2024,3540,102}},{"biscuits",{101,27529,1012,102}},{"soda biscuits",{101,14904,27529,102}},
        {"wafers",{101,11333,24396,102}},{"round biscuits",{101,2461,27529,102}},{"square biscuits",{101,2675,27529,102}},
        {"macaroons",{101,6097,10464,5644,102}}}; 

namespace yo{

struct Node{
    std::vector<int64_t> dim; // batch,channel,height,width
    char* name = nullptr;
};


class Model{
public:
    virtual ~Model(){};
    virtual int inference(cv::Mat &image) = 0;
    virtual int initialize(std::string onnx_path,bool is_cuda) = 0;
protected:
    virtual void preprocess(cv::Mat &image)=0;
    virtual void postprocess(std::vector<Ort::Value>& output_tensors)=0;
};

}