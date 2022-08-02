cmake_minimum_required(VERSION 3.5)
project(ji)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_BUILD_TYPE "Release")

#先编译三方库中的源码，并包含头文件路径
#jsoncpp相关头文件路径和源文件
include_directories(${CMAKE_SOURCE_DIR}/3rd/jsoncpp_simple/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd/jsoncpp_simple)
#配置点线框解析相关头文件路径和源文件
include_directories(${CMAKE_SOURCE_DIR}/3rd/wkt_parser/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd/wkt_parser)

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
set(ENV{PKG_CONFIG_PATH} "ENV{PKG_CONFIG_PATH}")
find_package(PkgConfig REQUIRED)

include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include/)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:/usr/local/ev_sdk/lib")

# 编译libji.so
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)
list(APPEND libji_SOURCES  ${SRCS})

add_library (${PROJECT_NAME} SHARED ${libji_SOURCES})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        glog
        WKTParser
        jsoncpp
        nvinfer
        nvonnxparser
        -L/usr/local/cuda/lib64/ -lcudart  
        )

# Install libji.so
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rd/fonts DESTINATION lib)
# ---  END EV_SDK DEPENDENCIES  ---