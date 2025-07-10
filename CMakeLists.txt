cmake_minimum_required(VERSION 3.16)

project(yolo11_rpi LANGUAGES CXX)


set(rknpu_yolo11_file rknpu2/yolo11.cc)


include_directories(./include)

add_library(rknnrt SHARED IMPORTED )

set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libs/librknnrt.so"
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

add_executable(yolo11_rpi main.cpp postprocess.cc ${rknpu_yolo11_file})


target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing
    rknnrt
    dl
)

include(GNUInstallDirs)
install(TARGETS yolo11_rpi
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
