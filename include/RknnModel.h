#ifndef RKNNMODEL_H
#define RKNNMODEL_H

#include <string>
#include <fstream>
#include <stdio.h>
#include <set>

// #include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include "postprocess.h"
#include "../utils/common.h"
#include "../utils/image_utils.h"



class RknnModel {

public:
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs = nullptr;
    rknn_tensor_attr* output_attrs = nullptr;
    int model_channel;
    int model_height;
    int model_width;
    bool is_quant;


    RknnModel();
    ~RknnModel();

    int InitModel(const char* model_filename);
    int PreProcess(const char*, image_buffer_t*, letterbox_t*);
    int PostProcess(rknn_output* outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
    int InferenceModel(const char*);
};


#endif // RKNNMODEL_H
