#include "yolo11.h"
#include <string>
#include <fstream>


struct rknn_model_context{
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;

};

class RknnModel {

public:
    rknn_model_context rknn_model_ctx;




    int InitModel(const char* model_filename){



        // read model from file
        std::ifstream fp(model_filename, std::ios::binary);

        if(!fp.is_open()){
            printf("Model file is not open");
            return 1;
        }

        fp.seekg (0, fp.end);
        int model_data_length = fp.tellg();
        fp.seekg (0, fp.beg);

        // allocate memory:
        char * model_data = new char [model_data_length];

        // read data as a block:
        fp.read(model_data,model_data_length);

        fp.close();

        //init

        rknn_context rknn_ctx;

        int ret = rknn_init(&rknn_ctx, model_data, model_data_length, RKNN_FLAG_COLLECT_PERF_MASK, NULL);

        if (!ret){
            printf("Error when initialize model");
            return 1;

        }

        delete[] model_data;


        // Получение данных о модели

        rknn_input_output_num model_in_out;

        ret = rknn_query(rknn_ctx,RKNN_QUERY_IN_OUT_NUM, &model_in_out, sizeof(rknn_input_output_num));

        if (!ret){
            printf("Error when get model input and output count");
            return 1;

        }

        rknn_tensor_attr input_attrs[model_in_out.n_input];

        for(int i=0; i<model_in_out.n_input;i++)
        {
            input_attrs[i].index = i;
            ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                             sizeof(rknn_tensor_attr));
        }


        rknn_tensor_attr output_attrs[model_in_out.n_output];

        for(int i=0; i<model_in_out.n_output;i++)
        {
            output_attrs[i].index = i;
            ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                             sizeof(rknn_tensor_attr));
        }


        if (!ret){
            printf("Error when get model input and output data");
            return 1;

        }

        rknn_model_ctx.rknn_ctx = rknn_ctx;
        rknn_model_ctx.io_num = model_in_out;
        rknn_model_ctx.input_attrs = input_attrs;
        rknn_model_ctx.output_attrs = output_attrs;

        return 0;
    }

        // PreProcess
        // Read Image

    int InferenceModel(){

        // PreProcess
        // Read Image




        return 0;
    }


private:



};
