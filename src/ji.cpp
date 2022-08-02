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
 * 示例代码：实现ji.h定义的sdk接口，
 *  开发者只需要对本示例中的少量代码进行修改即可:
 *      SampleAlgorithm.hpp-----修改算法实现的头文件名
 *      SampleAlgorithm---------修改算法实现的类型名称
 *      algo_version------------修改算法的版本信息
 * 请先浏览本demo示例的注释，帮助快速理解和进行开发
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <vector>

//算法自定义的头文件
#include "SampleAlgorithm.hpp"
//ev_sdk主要接口定义
#include "ji.h"

//算法版本号，开发者自定义，且要求每次版本更新后，按照一致的形式修改算法版本号
static const std::string algo_version = "1.0.1";

//ev_sdk的版本号，当前是4.0.0，不用开发者自定义，除非明确了采用新版的ev_sdk
static const std::string sdk_version = "4.0.0";

/**
* @brief 获取SDK和算法的版本信息,以json字符串的形式返回.
* @param pVersion 返回的版本的json字符串
* @return 返回SDK定义的返回码
**/
JiErrorCode ji_get_version(char *pVersion)
{
    JiErrorCode jRet = JISDK_RET_SUCCEED;
    if(pVersion == nullptr)
    {
        return JISDK_RET_FAILED;
    }
    std::string strVersionInfo = "{\"sdk_version\":\"" + sdk_version + "\"," + "\"algo_version\":\"" + algo_version + "\"}";
    SDKLOG(INFO) << "get sdk version info : " << strVersionInfo;
    if(MAX_VERSION_LENGTH < strVersionInfo.size() + 1)
    {
        SDKLOG(ERROR) << "version string is too long";
        jRet = JISDK_RET_FAILED;
    }    
    strncpy(pVersion, strVersionInfo.c_str(), min(int(strVersionInfo.size() + 1), MAX_VERSION_LENGTH));    
    return jRet;
}

JiErrorCode ji_init(int argc, char **argv)
{
    // 根据实际情况对SDK进行初始化，如授权功能
    return JISDK_RET_SUCCEED;
}

void ji_reinit()
{
    // 根据实际情况对sdk去初始化
    return;
}


/**
 * @brief 创建具体算法实例的接口，调用后会返回一个算法实例
 * @param pdtype 实例类型参数,暂时未使用该参数
 * @return 指向算法对象的指针
**/
void *ji_create_predictor(JiPredictorType pdtype) 
{
    auto *detector = new SampleAlgorithm;
    if (detector->Init() != SampleAlgorithm::STATUS_SUCCESS) 
    {
        delete detector;
        detector = nullptr;
        SDKLOG(ERROR) << "Predictor init failed.";
        return nullptr;
    }
    SDKLOG(INFO) << "SamplePredictor init OK.";
    return detector;
}

/**
 * @brief 销毁算法实例的接口，传入ji_create_predictor创建的算法实例,将其销毁
 * @param predictor 创建好的算法实例
**/
void ji_destroy_predictor(void *predictor) 
{
    if (predictor == nullptr) return;

    auto *detector = reinterpret_cast<SampleAlgorithm *>(predictor);//SampleAlgorithm为自定义的算法class类，需要根据情况进行类名修改！！！！
    detector->UnInit();
    delete detector;
    detector = nullptr;
}

/**
 * @brief ji_calc_image:算法同步分析接口，上层应用调用最主要的入口之一，具体入参说明可参见ev_sdk/doc/EV_SDK接口规范说明.md
 * 开发者需要根据自定义的算法class类进行修改，在该函数中主要是处理入参，并将相关参数传入算法class实例，并调用自定义的算法分析函数
 * 该接口需要注意点是：
 * 1、传入待分析的图像数据可以是单张图也可以是多张图（由nInCount指示，大多数情况算法每次分析只需要传入一张图，但也有部分情况算法分析需要同时结合多张图）
    
 * 2、自定义的算法需要处理该接口传入的args参数，该参数通常是跟配置相关的参数信息（比如roi、识别阈值以及任何算法支持改变的控制参数）,上层应用调用时会动态传入
     args参数，算法分析时需要动态适应。
* 3、输出的结果图（通常会存在需要将算法分析的一些结果信息显示到与原图相同大小的结果图上）放在pOutFrames中；同样地，输入几张图就对应几张输出图，数量由nOutCount指示。
* 4、算法输出的所有结构化信息（具体信息内容根据业务及算法决定，相关的一些基本规则可参见ev_sdk/doc/极市算法SDK输出协议.md）均以JSON规范封装在JiEvent数据结构中。
*  @param predictor 调用接口创建的算法实例
*  @param pInFrames 输入信息, 一般包含单张或多张图片数据
*  @param nInCount 输入图片的张数信息
*  @param args 算法输入的动态参数,roi等, 如果传入不为空的话需要动态更新
*  @param pOutFrames 输出图片信息
*  @param nOutCount 输出图片张数
*  @param event 输出结构体,包含json,报警信息等
*  @return 返回SDK定义的返回码
**/
JiErrorCode ji_calc_image(void* predictor, const JiImageInfo* pInFrames, const unsigned int nInCount, const char* args, 
						JiImageInfo **pOutFrames, unsigned int & nOutCount, JiEvent &event)
{
    if (predictor == nullptr || pInFrames == nullptr || nInCount <= 0) 
    {
        return JISDK_RET_INVALIDPARAMS;
    }

    auto *detector = reinterpret_cast<SampleAlgorithm *>(predictor);//SampleAlgorithm为自定义的算法class类，需要根据情况进行类名修改！！！！
    STATUS processRet;
    // 例： 分析单张图片，输出单张图片，算法内部使用cv::Mat格式数据
    if(nInCount == 1)
    {
        cv::Mat inMat(pInFrames[0].nHeight, pInFrames[0].nWidth, CV_8UC3, pInFrames[0].pData);//获取到需要进行分析的图像数据
        processRet = detector->Process(inMat, args, event);//算法进行分析
    }
    else
    {
        /* 多图自行实现,如：
        for(unsigned int i = 0 ; i < nInCount; i++)
        {
            ...
        }
        */
    }    
    if (SampleAlgorithm::STATUS_SUCCESS == processRet)
    {
        if (event.code != JISDK_CODE_FAILED)
        {
            detector->GetOutFrame(pOutFrames,nOutCount);//算法处理输出结果图
            return JISDK_RET_SUCCEED;
        }
        else
        {
            return JISDK_RET_FAILED;
        }
    }
    return JISDK_RET_FAILED;
}

/**
* @brief 算法配置参数更新接口，上层应用显式调用算法配置更新
* @param predictor 创建的算法实例指针
* @param args 参数字符串,json形式组织
* @return 返回SDK定义的返回码
**/
JiErrorCode ji_update_config(void *predictor, const char *args)
{
    if(args == nullptr)
    {
        SDKLOG(ERROR) << "config string is null ";
        return JISDK_RET_FAILED;
    }
    SDKLOG(INFO) << "update : " << args;
    auto *detector = reinterpret_cast<SampleAlgorithm *>(predictor);//SampleAlgorithm为自定义的算法class类，需要根据情况进行类名修改！！！！

    if(detector->UpdateConfig(args) != SampleAlgorithm::STATUS_SUCCESS)//调用自定义算法class类的配置update函数
        return JISDK_RET_FAILED;
    
    return JISDK_RET_SUCCEED;
}


/*
* ji_set_callback:算法异步分析相关的callback设置接口
* 在本demo中未实现，可具体根据项目和算法要求进行相关实现!!!!
*/
JiErrorCode ji_set_callback(void *predictor, JiCallBack callback)
{
    return JISDK_RET_UNUSED;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/*
* ji_calc_image_asyn:算法异步分析接口
* 在本demo中未实现，可具体根据项目和算法要求进行相关实现!!!!
*/
JiErrorCode ji_calc_image_asyn(void* predictor, const JiImageInfo* pInFrames, const unsigned int nInCount, const char* args, void *userData)
{
    return JISDK_RET_UNUSED;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/*
* ji_face_*:与人脸识别相关的接口
* 在本demo中未实现，可具体根据项目和算法要求进行相关实现!!!!
*/

JiErrorCode ji_create_face_db(void *predictor, const char *faceDBName, const int faceDBId, const char *faceDBDes)
{
    return JISDK_RET_UNUSED;
}


JiErrorCode ji_delete_face_db(void *predictor, const int faceDBId)
{
    return JISDK_RET_UNUSED;
}


JiErrorCode ji_get_face_db_info(void *predictor, const int faceDBId, char *info)
{
    return JISDK_RET_UNUSED;
}


JiErrorCode ji_face_add(void *predictor, const int faceDBId, const char *faceName, const int faceId, const char *data, const int dataType, char *imagePath)
{
    return JISDK_RET_UNUSED;
}


JiErrorCode ji_face_update(void *predictor, const int faceDBId, const char *faceName, const int faceId, const char *data, const int dataType, char *imagePath)
{
    return JISDK_RET_UNUSED;
}


JiErrorCode ji_face_delete(void *predictor, const int faceDBId, const int faceId)
{
    return JISDK_RET_UNUSED;
}