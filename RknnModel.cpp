#include "RknnModel.h"
#include <iostream>
#include <algorithm>
#include <cstring>

RknnModel::RknnModel(){

    rknn_ctx = 0;
    io_num = {};
    input_attrs = nullptr;
    output_attrs = nullptr;
    model_channel = 0;
    model_height = 0;
    model_width = 0;
    is_quant = true;


}


RknnModel::~RknnModel(){
    if (rknn_ctx != 0){

        rknn_destroy(rknn_ctx);
        rknn_ctx = 0;
    }
    delete[] input_attrs;
    delete[] output_attrs;

}


int RknnModel::InitModel(const char* model_filename){
    // read model from file
    std::ifstream fp(model_filename, std::ios::binary);

    if(!fp.is_open()){
        printf("Model file is not open\n");
        return 1;
    }

    fp.seekg (0, fp.end);
    int model_data_length = fp.tellg();
    fp.seekg (0, fp.beg);

    // allocate memory:
    char* model_data = new char [model_data_length];

    // read data as a block:
    fp.read(model_data,model_data_length);

    if (fp)
        std::cout << "all characters read successfully." << "\n";
    else
        std::cout << "error: only " << fp.gcount() << " could be read" << "\n";


    fp.close();

    printf("model_data_length %d\n", model_data_length);



    //init

    rknn_context rknn_ctx;

    int ret = rknn_init(&rknn_ctx, model_data, model_data_length, RKNN_FLAG_COLLECT_PERF_MASK, NULL);;

    delete[] model_data;

    if (ret){

        printf("Error when initialize model %d\n", ret);
        rknn_destroy(rknn_ctx);
        this->~RknnModel();
        return 2;

    }
    // Получение данных о модели

    rknn_input_output_num model_in_out;

    ret = rknn_query(rknn_ctx,RKNN_QUERY_IN_OUT_NUM, &model_in_out, sizeof(rknn_input_output_num));

    if (ret){
        printf("Error when get model input and output count");
        return 3;

    }

    input_attrs = new rknn_tensor_attr[model_in_out.n_input];
    output_attrs = new rknn_tensor_attr[model_in_out.n_output];



    for(int i=0; i<model_in_out.n_input;i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("Error querying input %d attribute: ret = %d\n", i, ret);
            return 4; // Новая ошибка для выходов
        }
    }


    for(int i = 0; i < model_in_out.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("Error querying output %d attribute: ret = %d\n", i, ret);
            return 5; // Новая ошибка для выходов
        }
    }


    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }

    printf("model input height=%d, width=%d, channel=%d\n",
           model_height, model_width, model_channel);


    this->rknn_ctx = rknn_ctx;
    this->io_num = model_in_out;
    this->input_attrs = input_attrs;
    this->output_attrs = output_attrs;

    return 0;
}

int RknnModel::PreProcess(const char*  image_path, image_buffer_t* dst_img, letterbox_t* letter_box){

    int ret;
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));



    ret = read_image(image_path, &src_image);



    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    int bg_color = 114;

    if ((!rknn_ctx) || !(&src_image))
    {
        return -1;
    }


    memset(letter_box, 0, sizeof(letterbox_t));
    memset(dst_img, 0, sizeof(image_buffer_t));

    // Pre Process
    dst_img->width = model_width;
    dst_img->height = model_height;
    dst_img->format = IMAGE_FORMAT_RGB888;
    dst_img->size = get_image_size(dst_img);
    dst_img->virt_addr = (unsigned char *)malloc(dst_img->size);
    if (dst_img->virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img->size);
        return -1;
    }

    ret = convert_image_with_letterbox(&src_image, dst_img, letter_box, bg_color);


    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    return 0;

}

int RknnModel::PostProcess(rknn_output* _outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results){

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = model_width;
    int model_in_h = model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));


    int dfl_len = output_attrs[0].dims[1] / 4;

    if (io_num.n_output > 0) {
        int dim_index = (output_attrs[0].n_dims == 4) ? 1 : 0;
        dfl_len = output_attrs[0].dims[dim_index] / 4;
    }


    int output_per_branch = io_num.n_output / 3;
    for (int i = 0; i < 3; i++)
    {

        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = _outputs[i*output_per_branch + 2].buf;
            score_sum_zp = output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;


        grid_h = output_attrs[box_idx].dims[2];
        grid_w = output_attrs[box_idx].dims[3];

        stride = model_in_h / grid_h;

        if (is_quant)
        {

            validCount += process_i8((int8_t *)_outputs[box_idx].buf, output_attrs[box_idx].zp, output_attrs[box_idx].scale,
                                     (int8_t *)_outputs[score_idx].buf, output_attrs[score_idx].zp, output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, conf_threshold);

        }
        else
        {
            validCount += process_fp32((float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len,
                                       filterBoxes, objProbs, classId, conf_threshold);
        }

    }

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;

}

int RknnModel::InferenceModel(const char*  image_path){
    int ret;

    image_buffer_t dst_img;
    letterbox_t letter_box;

    PreProcess(image_path, &dst_img, &letter_box);

    rknn_input inputs[io_num.n_input];
    rknn_output outputs[io_num.n_output];
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;

    memset(outputs, 0, sizeof(outputs));

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = model_width * model_height * model_channel;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(rknn_ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!is_quant);
    }


    ret = rknn_outputs_get(rknn_ctx, io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);

    }

    object_detect_result_list od_results;

    PostProcess(outputs, &letter_box, box_conf_threshold, nms_threshold, &od_results);

    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%d @ (%d %d %d %d) %.3f\n", det_result->cls_id,
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);



    }

    rknn_perf_detail perf_detail;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));

    std::ofstream out;          // поток для записи
    out.open("yolo_report1.txt");      // открываем файл для записи
    if (out.is_open())
    {
        out << perf_detail.perf_data << std::endl;
    }
    out.close();

    rknn_outputs_release(rknn_ctx, io_num.n_output, outputs);

    return 0;
}


