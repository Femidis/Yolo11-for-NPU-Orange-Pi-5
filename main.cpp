#include "RknnModel.h"


int main(int argc, char **argv)
{
    // if (argc != 2)
    // {
    //     printf("%s <model_path>\n", argv[0]);
    //     return -1;
    // }



    // const char *model_path = argv[1];

    const char *model_path = "/home/ab/yolo11_rpi/model/yolo11n.rknn";

    // const char *image_path = argv[2];

    int ret = 0;

    RknnModel current_model;

    ret = current_model.InitModel(model_path);

    if (ret){
        printf("Initialization fault with code: %d\n", ret);
        return 1;
    }

    if (current_model.rknn_ctx == 0) {
        printf("Ошибка: контекст уже уничтожен или не инициализирован\n");
        return 1;
    }

    ret = current_model.InferenceModel("/home/ab/yolo11_rpi/model/bus.jpg");

    printf("Finish\n");

    return 0;
}
