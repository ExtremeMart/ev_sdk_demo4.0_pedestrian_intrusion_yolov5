/*
 * Copyright (c) 2021 ExtremeVision Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
* 本demo采用YOLO5目标检测器,检测车辆和车牌
* 首次运行时采用tensorrt先加载onnx模型,并保存trt模型,以便下次运行时直接加载trt模型,加快初始化
*
*/

#ifndef COMMON_DET_INFER_H
#define COMMON_DET_INFER_H
#include <memory>
#include <map>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include "opencv2/core.hpp"

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static float IOU(const cv::Rect& b1, const cv::Rect& b2)
{
    auto intersec = b1 & b2;
    return static_cast<float>(intersec.area()) / ( b1.area() + b2.area() - intersec.area() );
}

class SampleDetector
{            
    public:
        SampleDetector();
        ~SampleDetector();
        bool Init(const std::string& strModelName, float thresh);
        bool UnInit();        
        bool ProcessImage(const cv::Mat& img, std::vector< BoxInfo >& DetObj, float thresh = 0.15);                   
        static void runNms(std::vector<BoxInfo>& objects, float thresh);
    private:
        void loadOnnx(const std::string strName);
        void loadTrt(const std::string strName);        
        void decode_outputs(float* prob, float thresh, std::vector<BoxInfo>& objects, float scale, const int img_w, const int img_h);
        
    
    private:    
        nvinfer1::ICudaEngine *m_CudaEngine; 
        nvinfer1::IRuntime *m_CudaRuntime;
        nvinfer1::IExecutionContext *m_CudaContext;
        cudaStream_t m_CudaStream;
        void* m_ArrayDevMemory[2]{0};
        void* m_ArrayHostMemory[2]{0};
        int m_ArraySize[2]{0};
        int m_iInputIndex;
        int m_iOutputIndex;        
        int m_iClassNums;
        int m_iBoxNums;
        cv::Size m_InputSize;
        cv::Mat m_Resized;
        cv::Mat m_Normalized;
        std::vector<cv::Mat> m_InputWrappers{};        
    
    private:
        bool m_bUninit = false;
        float mThresh;
};



#endif 