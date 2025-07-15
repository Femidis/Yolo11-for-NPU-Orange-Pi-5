cmake_minimum_required(VERSION 3.16)

project(yolo11_rpi LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")  # Linux: $ORIGIN означает папку с исполняемым файлом
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(${CMAKE_INSTALL_RPATH})


set(CMAKE_INSTALL_PREFIX /home/ab/yolo11_rpi)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_library(rknnrt SHARED IMPORTED)

set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libs/librknnrt.so"
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

# set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-aarch64/share/")
# find_package(OpenCV REQUIRED)



add_executable(yolo11_rpi

    RknnModel.cpp
    main.cpp
    include/RknnModel.h
    postprocess.cc

)



target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing
    rknnrt
    dl
)



include(GNUInstallDirs)

# link_directories(${CMAKE_INSTALL_LIBDIR})


install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/libs/librknnrt.so DESTINATION ${CMAKE_INSTALL_LIBDIR})


install(TARGETS yolo11_rpi
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
