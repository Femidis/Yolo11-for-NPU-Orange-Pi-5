#ifndef _RKNN_YOLO11_DEMO_POSTPROCESS_H_
#define _RKNN_YOLO11_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "../utils/common.h"
#include "../utils/image_utils.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

// class rknn_app_context_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
               int8_t *score_tensor, int32_t score_zp, float score_scale,
               int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
               int grid_h, int grid_w, int stride, int dfl_len,
               std::vector<float> &boxes,
               std::vector<float> &objProbs,
               std::vector<int> &classId,
               float threshold);

int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor,
                 int grid_h, int grid_w, int stride, int dfl_len,
                 std::vector<float> &boxes,
                 std::vector<float> &objProbs,
                 std::vector<int> &classId,
                 float threshold);

int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);

int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
        int filterId, float threshold);

int clamp(float val, int min, int max);


// int init_post_process();
// void deinit_post_process();
// char *coco_cls_to_name(int cls_id);
// int post_process(RknnModel *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);

// void deinitPostProcess();
#endif //_RKNN_YOLO11_DEMO_POSTPROCESS_H_
