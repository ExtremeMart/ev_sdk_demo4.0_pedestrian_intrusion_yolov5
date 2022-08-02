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

#ifndef JI_ALGORITHM_CONFIGURATION
#define JI_ALGORITHM_CONFIGURATION
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "WKTParser.h"
#include "ji_utils.h"
#include "reader.h"
#include "writer.h"
#include "value.h"

#define BGRA_CHANNEL_SIZE 4

typedef struct
{
    // 与自定义算法密切相关的配置参数，可封装在该结构体中
    double thresh;
} AlgoConfig;

/**
 * @brief 存储和更新配置参数
 */
struct Configuration
{
    private:
    Json::Value mJConfigValue;
    public:    
    // 算法与画图的可配置参数及其默认值
    // 1. roi配置
    std::vector<cv::Rect> currentROIRects;           // 多边形roi区域对应的矩形区域（roi可能是多个）
    std::vector<VectorPoint> currentROIOrigPolygons; // 原始的多边形roi区域（roi可能是多个）
    std::vector<std::string> origROIArgs;            //原始的多边形roi区域对应的字符串
    cv::Size currentInFrameSize{0, 0};               // 当前处理帧的尺寸
    // 2. 与ROI显示相关的配置
    bool drawROIArea = false;                    // 是否画ROI
    cv::Scalar roiColor = {120, 120, 120, 1.0f}; // ROI框的颜色
    int roiLineThickness = 4;                    // ROI框的粗细
    bool roiFill = false;                        // 是否使用颜色填充ROI区域
    bool drawResult = true;                      // 是否画识别结果
    bool drawConfidence = false;                 // 是否画置信度
    // --------------------------------- 通常需要根据需要修改 START -----------------------------------------
    // 3. 算法配置参数
    AlgoConfig algoConfig = {0.4}; // 默认的算法配置
    // 4. 与报警信息相关的配置
    std::string language = "en";                             // 所显示文字的默认语言
    int targetRectLineThickness = 4;                         // 目标框粗细
    std::map<std::string, std::vector<std::string> > targetRectTextMap = { {"en",{"vehicle", "plate"}}, {"zh", {"车辆","车牌"}}};// 检测目标框顶部文字
    cv::Scalar targetRectColor = {0, 255, 0, 1.0f}; // 检测框`mark`的颜色
    cv::Scalar textFgColor = {0, 0, 0, 0};          // 检测框顶部文字的颜色
    cv::Scalar textBgColor = {255, 255, 255, 0};    // 检测框顶部文字的背景颜色
    int targetTextHeight = 30;                      // 目标框顶部字体大小

    bool drawWarningText = true;
    int warningTextSize = 40;                             // 画到图上的报警文字大小
    std::map<std::string, std::string> warningTextMap = { {"en", "WARNING! WARNING!"}, {"zh", "警告"}};// 画到图上的报警文字
    cv::Scalar warningTextFg = {255, 255, 255, 0}; // 报警文字颜色
    cv::Scalar warningTextBg = {0, 0, 255, 0};     // 报警文字背景颜色
    cv::Point warningTextLeftTop{0, 0};            // 报警文字左上角位置
    // --------------------------------- 通常需要根据需要修改 END -------------------------------------------
    //解析数值类型的配置
    template <typename T>
    bool checkAndUpdateNumber(const std::string& key, T &val)
    {
      return  mJConfigValue.isMember(key) && mJConfigValue[key].isNumeric() && ( (val = mJConfigValue[key].asDouble()) || true); 
    }
    //解析字符串类型配置的函数
    bool checkAndUpdateStr(const std::string& key, std::string &val)
    {
        return mJConfigValue.isMember(key) && mJConfigValue[key].isString() &&  (val = mJConfigValue[key].asString()).size(); 
    }
    //解析字符串数组类型配置的函数
    bool checkAndUpdateVecStr(const std::string& key, std::vector<std::string> &val)
    {
        if( mJConfigValue.isMember(key) && mJConfigValue[key].isArray() ) 
        {
            val.resize(mJConfigValue[key].size());
            for(int i = 0; i <  mJConfigValue[key].size(); ++i)
            {
                val[i] = mJConfigValue[key][i].asString();
            }
        }
    }
    //解析bool类型配置的函数
    bool checkAndUpdateBool(const std::string& key, bool &val)
    {
        return mJConfigValue.isMember(key) && mJConfigValue[key].isBool() && ( (val = mJConfigValue[key].asBool()) || true); 
    }
    //解析颜色配置的函数
    bool checkAndUpdateColor(const std::string& key, cv::Scalar &color)
    {
        if(mJConfigValue.isMember(key) && mJConfigValue[key].isArray() && mJConfigValue[key].size() == BGRA_CHANNEL_SIZE)
        {
            for (int i = 0; i < BGRA_CHANNEL_SIZE; ++i)
            {
                color[i] = mJConfigValue[key][i].asDouble();
            }
            return true;
        }
        return false;        
    }
    //解析点配置的函数
    bool checkAndUpdatePoint(const std::string& key, cv::Point &point)
    {
        if(mJConfigValue.isMember(key) && mJConfigValue[key].isArray() && mJConfigValue[key].size() == 2 && mJConfigValue[key][0].isNumeric() && mJConfigValue[key][1].isNumeric())
        {
            point = cv::Point(mJConfigValue[key][0].asDouble(), mJConfigValue[key][1].asDouble());
            return true;
        }
        return false;        
    }

    /**
     * @brief 解析json格式的配置参数,是开发者需要重点关注和修改的地方！！！     
     * @param[in] configStr json格式的配置参数字符串
     * @return 当前参数解析后，生成的算法相关配置参数
     */
    void ParseAndUpdateArgs(const char *confStr)
    {
        if (confStr == nullptr)
        {
            SDKLOG(INFO) << "Input is none";
            return;
        }        
        Json::Reader reader;
        if( !reader.parse(confStr, mJConfigValue) )
        {
            SDKLOG(ERROR) << "failed to parse config " << confStr;
        }
        checkAndUpdateBool("draw_roi_area", drawROIArea);
        checkAndUpdateNumber("thresh", algoConfig.thresh);            
        checkAndUpdateNumber("roi_line_thickness", roiLineThickness);
        checkAndUpdateBool("roi_fill", roiFill);
        checkAndUpdateStr("language", language);
        checkAndUpdateBool("draw_result", drawResult);
        checkAndUpdateBool("draw_confidence", drawConfidence);
        checkAndUpdateVecStr("mark_text_en", targetRectTextMap["en"]);        
        checkAndUpdateVecStr("mark_text_zh", targetRectTextMap["zh"]);        
        checkAndUpdateColor("roi_color", roiColor);
        checkAndUpdateColor("object_text_color", textFgColor);
        checkAndUpdateColor("object_text_bg_color", textBgColor);
        checkAndUpdateColor("target_rect_color", targetRectColor);
        checkAndUpdateNumber("object_rect_line_thickness", targetRectLineThickness);
        checkAndUpdateNumber("object_text_size", targetTextHeight);
        checkAndUpdateBool("draw_warning_text", drawWarningText);
        checkAndUpdateNumber("warning_text_size", warningTextSize);
        checkAndUpdateStr("warning_text_en", warningTextMap["en"]);  
        checkAndUpdateStr("warning_text_zh", warningTextMap["zh"]);  
        checkAndUpdateColor("warning_text_color", warningTextFg);
        checkAndUpdateColor("warning_text_bg_color", warningTextBg);       
        checkAndUpdatePoint("warning_text_left_top", warningTextLeftTop);
        std::vector<std::string> roiStrs;
        if(mJConfigValue.isMember("polygon_1") && mJConfigValue["polygon_1"].isArray() && mJConfigValue["polygon_1"].size() )
        {
            for (int i = 0; i < mJConfigValue["polygon_1"].size(); ++i)
            {                
                if(mJConfigValue["polygon_1"][i].isString())
                {
                    roiStrs.emplace_back(mJConfigValue["polygon_1"][i].asString());
                }
            }
        }        
        if (!roiStrs.empty())
        {
            origROIArgs = roiStrs;
            UpdateROIInfo(currentInFrameSize.width, currentInFrameSize.height);//根据当前输入图像帧的大小更新roi参数
        }
                
        return;
    }
    /**
     * @brief 当输入图片尺寸变更时，更新ROI
     **/
    void UpdateROIInfo(int newWidth, int newHeight)
    {
        currentInFrameSize.width = newWidth;
        currentInFrameSize.height = newHeight;
        currentROIOrigPolygons.clear();
        currentROIRects.clear();

        VectorPoint currentFramePolygon;
        currentFramePolygon.emplace_back(cv::Point(0, 0));
        currentFramePolygon.emplace_back(cv::Point(currentInFrameSize.width, 0));
        currentFramePolygon.emplace_back(cv::Point(currentInFrameSize.width, currentInFrameSize.height));
        currentFramePolygon.emplace_back(cv::Point(0, currentInFrameSize.height));

        WKTParser wktParser(cv::Size(newWidth, newHeight));
        for (auto &roiStr : origROIArgs)
        {
            SDKLOG(INFO) << "parsing roi:" << roiStr;
            VectorPoint polygon;
            wktParser.parsePolygon(roiStr, &polygon);
            bool isPolygonValid = true;
            for (auto &point : polygon)
            {
                if (!wktParser.inPolygon(currentFramePolygon, point))
                {
                    SDKLOG(ERROR) << "point " << point << " not in polygon!";
                    isPolygonValid = false;
                    break;
                }
            }
            if (!isPolygonValid || polygon.empty())
            {
                SDKLOG(ERROR) << "roi `" << roiStr << "` not valid! skipped!";
                continue;
            }
            currentROIOrigPolygons.emplace_back(polygon);
        }
        if (currentROIOrigPolygons.empty())
        {
            currentROIOrigPolygons.emplace_back(currentFramePolygon);
            SDKLOG(WARNING) << "Using the whole image as roi!";
        }

        for (auto &roiPolygon : currentROIOrigPolygons)
        {
            cv::Rect rect;
            wktParser.polygon2Rect(roiPolygon, rect);
            currentROIRects.emplace_back(rect);
        }
    }
};
#endif
