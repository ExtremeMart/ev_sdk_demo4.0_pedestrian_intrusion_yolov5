#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

#include "reader.h"
#include "writer.h"
#include "value.h"
#include "ji_utils.h"
#include "SampleAlgorithm.hpp"

#define JSON_ALERT_FLAG_KEY ("is_alert")
#define JSON_ALERT_FLAG_TRUE true
#define JSON_ALERT_FLAG_FALSE false


SampleAlgorithm::SampleAlgorithm()
{    
}

SampleAlgorithm::~SampleAlgorithm()
{    
    UnInit();
}

STATUS SampleAlgorithm::Init()
{
    // 从默认的配置文件读取相关配置参数
    const char *configFile = "/usr/local/ev_sdk/config/algo_config.json";
    SDKLOG(INFO) << "Parsing configuration file: " << configFile;
    std::ifstream confIfs(configFile);
    if (confIfs.is_open())
    {
        size_t len = getFileLen(confIfs);
        char *confStr = new char[len + 1];
        confIfs.read(confStr, len);
        confStr[len] = '\0';
    	SDKLOG(INFO) << "Configs:"<<confStr;
        mConfig.ParseAndUpdateArgs(confStr);
        delete[] confStr;
        confIfs.close();
    }
    mDetector = std::make_shared<SampleDetector>();
    mDetector->Init("/usr/local/ev_sdk/model/yolov5s.onnx", mConfig.algoConfig.thresh);
    return STATUS_SUCCESS;
}


STATUS SampleAlgorithm::UnInit()
{
    if(mDetector.get() != nullptr)
    {
        SDKLOG(INFO) << "uninit";
        mDetector->UnInit();
        mDetector.reset();
    }
    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::UpdateConfig(const char *args)
{
    if (args == nullptr)
    {
        SDKLOG(ERROR) << "mConfig string is null ";
        return ERROR_CONFIG;
    }
    mConfig.ParseAndUpdateArgs(args);
    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::GetOutFrame(JiImageInfo **out, unsigned int &outCount)
{
    outCount = mOutCount;

    mOutImage[0].nWidth = mOutputFrame.cols;
    mOutImage[0].nHeight = mOutputFrame.rows;
    mOutImage[0].nFormat = JI_IMAGE_TYPE_BGR;
    mOutImage[0].nDataType = JI_UNSIGNED_CHAR;
    mOutImage[0].nWidthStride = mOutputFrame.step;
    mOutImage[0].pData = mOutputFrame.data;

    *out = mOutImage;
    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::Process(const cv::Mat &inFrame, const char *args, JiEvent &event)
{    
    //输入图片为空的时候直接返回错误
    if (inFrame.empty())
    {
        SDKLOG(ERROR) << "Invalid input!";
        return ERROR_INPUT;
    }
   
    //由于roi配置是归一化的坐标,所以输出图片的大小改变时,需要更新ROI的配置  
    if (inFrame.cols != mConfig.currentInFrameSize.width || inFrame.rows != mConfig.currentInFrameSize.height)
    {
	    SDKLOG(INFO)<<"Update ROI Info...";
        mConfig.UpdateROIInfo(inFrame.cols, inFrame.rows);
    }

    //如果输入的参数不为空且与上一次的参数不完全一致,需要调用更新配置的接口
    if(args != nullptr && mStrLastArg != args) 
    {	
    	mStrLastArg = args;
        SDKLOG(INFO) << "Update args:" << args;
        mConfig.ParseAndUpdateArgs(args);
    }

    // 针对整张图进行推理,获取所有的检测目标,并过滤出在ROI内的目标
    std::vector<BoxInfo> detectedObjects;    
    std::vector<BoxInfo> validTargets;
    // 算法处理
    cv::Mat img = inFrame.clone();
    mDetector->ProcessImage(img, detectedObjects, mConfig.algoConfig.thresh);    
    
    //过滤出行人
    for(auto iter = detectedObjects.begin(); iter != detectedObjects.end();)
    {
        if(iter->label ==0 )
        {
            iter++;
        }
        else
        {
            iter = detectedObjects.erase(iter);
        }
    }
    
    
    for (auto &obj : detectedObjects)
    {
        for (auto &roiPolygon : mConfig.currentROIOrigPolygons)
        {
            int mid_x = (obj.x1 + obj.x2) / 2;
            int mid_y = (obj.y1 + obj.y2) / 2;
            // 当检测的目标的中心点在ROI内的话，就视为闯入ROI的有效目标
            if (WKTParser::inPolygon(roiPolygon, cv::Point(mid_x, mid_y)))
            {
                validTargets.emplace_back(obj);
            }           
        }
    }
    SDKLOG_FIRST_N(INFO, 5) << "detected targets : " << detectedObjects.size() << " valid targets :  " << validTargets.size();
    // 此处示例业务逻辑：当算法检测到有行人闯入时，就报警
    bool isNeedAlert = false; // 是否需要报警
    // 创建输出图
    inFrame.copyTo(mOutputFrame);
    // 画ROI区域
    if (mConfig.drawROIArea && !mConfig.currentROIOrigPolygons.empty())
    {
        drawPolygon(mOutputFrame, mConfig.currentROIOrigPolygons, cv::Scalar(mConfig.roiColor[0], mConfig.roiColor[1], mConfig.roiColor[2]),
                    mConfig.roiColor[3], cv::LINE_AA, mConfig.roiLineThickness, mConfig.roiFill);
    }
    // 判断是否要要报警
    if (validTargets.size() > 0)
    {        
        isNeedAlert = true;
    }
    //并将检测到的在ROI内部的目标画到图上
    for (auto &object : validTargets)
    {
        if (mConfig.drawResult)
        {
            std::stringstream ss;
            ss << (object.label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "": mConfig.targetRectTextMap[mConfig.language][object.label]);
            if (mConfig.drawConfidence)
            {
                ss.precision(0);
                ss << std::fixed << (object.label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "" : ": ") << object.score * 100 << "%";
            }
            cv::Rect rect = cv::Rect{object.x1, object.y1, object.x2 - object.x1, object.y2 - object.y1};
            drawRectAndText(mOutputFrame, rect, ss.str(), mConfig.targetRectLineThickness, cv::LINE_AA,
                            cv::Scalar(mConfig.targetRectColor[0], mConfig.targetRectColor[1], mConfig.targetRectColor[2]), mConfig.targetRectColor[3], mConfig.targetTextHeight,
                            cv::Scalar(mConfig.textFgColor[0], mConfig.textFgColor[1], mConfig.textFgColor[2]),
                            cv::Scalar(mConfig.textBgColor[0], mConfig.textBgColor[1], mConfig.textBgColor[2]));
        }
    }

    if (isNeedAlert && mConfig.drawWarningText)
    {
        drawText(mOutputFrame, mConfig.warningTextMap[mConfig.language], mConfig.warningTextSize,
                 cv::Scalar(mConfig.warningTextFg[0], mConfig.warningTextFg[1], mConfig.warningTextFg[2]),
                 cv::Scalar(mConfig.warningTextBg[0], mConfig.warningTextBg[1], mConfig.warningTextBg[2]), mConfig.warningTextLeftTop);
    }

    // 将结果封装成json字符串
    bool jsonAlertCode = JSON_ALERT_FLAG_FALSE;
    if (isNeedAlert)
    {
        jsonAlertCode = JSON_ALERT_FLAG_TRUE;
    }
    Json::Value jRoot;
    Json::Value jAlgoValue;
    Json::Value jDetectValue;
    
    jAlgoValue[JSON_ALERT_FLAG_KEY] = jsonAlertCode;    
    jAlgoValue["target_info"].resize(0);
    for (auto &obj : validTargets)
    {        
        Json::Value jTmpValue;
        jTmpValue["x"] = int(obj.x1);
        jTmpValue["y"] = int(obj.y1);
        jTmpValue["width"] = int(obj.x2 - obj.x1);
        jTmpValue["height"] = int(obj.y2 - obj.y1);
        jTmpValue["name"] = (obj.label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "obj": mConfig.targetRectTextMap[mConfig.language][obj.label]);
        jTmpValue["confidence"] = obj.score;
        jAlgoValue["target_info"].append(jTmpValue);
    }
    jRoot["algorithm_data"] = jAlgoValue;
    
    //create model data
    jDetectValue["objects"].resize(0);   
    for (auto &obj : detectedObjects)
    {
        Json::Value jTmpValue;
        jTmpValue["x"] = int(obj.x1);
        jTmpValue["y"] = int(obj.y1);
        jTmpValue["width"] = int(obj.x2 - obj.x1);
        jTmpValue["height"] = int(obj.y2 - obj.y1);
        jTmpValue["name"] = (obj.label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "obj": mConfig.targetRectTextMap[mConfig.language][obj.label]);
        jTmpValue["confidence"] = obj.score;
        jDetectValue["objects"].append(jTmpValue);
    }
    jRoot["model_data"] = jDetectValue;

    Json::StreamWriterBuilder writerBuilder;
    writerBuilder.settings_["precision"] = 2;
    writerBuilder.settings_["emitUTF8"] = true;    
    std::unique_ptr<Json::StreamWriter> jsonWriter(writerBuilder.newStreamWriter());
    std::ostringstream os;
    jsonWriter->write(jRoot, &os);
    mStrOutJson = os.str();    
    // 注意：JiEvent.code需要根据需要填充，切勿弄反
    if (isNeedAlert)
    {
        event.code = JISDK_CODE_ALARM;
    }
    else
    {
        event.code = JISDK_CODE_NORMAL;
    }
    event.json = mStrOutJson.c_str();    
    return STATUS_SUCCESS;
}

