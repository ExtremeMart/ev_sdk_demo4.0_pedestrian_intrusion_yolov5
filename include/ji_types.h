/*
 * Copyright (c) 2021 Extreme Vision Co., Ltd.
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

#ifndef EV_JI_TYPES_H_
#define EV_JI_TYPES_H_

/**
 * @brief Enumerates predictor type codes.
 *
 * @since 4.0
 */
typedef enum {
    /** Default. */
    JISDK_PREDICTOR_DEFAULT = 0,

    /** Continuous predictor(predictor with status). */
    JISDK_PREDICTOR_SEQUENTIAL = 1,

    /** Non-continuous predictor(predictor without status). */
    JISDK_PREDICTOR_NONSEQUENTIAL = 2
} JiPredictorType;

/**
 * @brief Enumerates image format type codes.
 *
 * @since 4.0
 */
typedef enum {
    /** Picture in BGR format. */
    JI_IMAGE_TYPE_BGR = 0,

    /** Picture in YUV420 format. */
    JI_IMAGE_TYPE_YUV420 = 1,

    /** Picture in YUV422 format. */
    JI_IMAGE_TYPE_YUV442 = 2
} JiImageFormat;

/**
 * @brief Enumerates the codes of data type.
 *
 * @since 4.0
 */
typedef enum {
    /** uint32_t type. */
    JI_UINT32_T = 0,

    /** signed char type. */
    JI_SIGNED_CHAR = 1,

    /** unsigned char type. */
    JI_UNSIGNED_CHAR = 2,

    /** unsigned short type. */
    JI_UNSIGNED_SHORT = 3,

    /** int64_t type. */
    JI_INT64_T = 4,

    /** uint64_t type. */
    JI_UINT64 = 5
} JiDataType;

/**
 * @brief Represents the info of image.
 *
 * @since 4.0
 */
typedef struct {
    /** Width of image. */
    unsigned int    nWidth;

    /** Height of image. */
    unsigned int    nHeight;

    /** Stride of image width.  */
    unsigned int    nWidthStride;

    /** Stride of image height. */
    unsigned int    nHeightStride;

    /** Sampling rate. */
    unsigned int    nFrameRate;

    /** Timestamp of image. */
    unsigned long   dwTimeStamp;

    /** Date of image. */
    void*           pData; 

    /** Data length of image. */
    unsigned int    nDataLen; 

    /** Image format. */
    JiImageFormat nFormat;

    /** date type. */
    JiDataType    nDataType;

    /** Frame number. */
    unsigned int    nFrameNo;

    /** Reserved. */
    unsigned char   byRes[4];
} JiImageInfo;


/**
 * @brief Enumerates event codes.
 *
 * @since 4.0
 */
typedef enum {
    /** type of alarm event. */
    JISDK_CODE_ALARM = 0,

    /** type of normal event. */
    JISDK_CODE_NORMAL = 1,

    /** type of invalid event. */
    JISDK_CODE_FAILED = -1
} JiEventTye;


/**
 * @brief Represents algorithm processing result events.
 *
 * @since 4.0
 */
typedef struct {
    /** type of event. */
	JiEventTye code;

    /** content of event. */
	const char * json;
} JiEvent;

typedef struct
{
    JiImageInfo outPic;
    JiEvent event;
}JiData;

typedef void (* JiCallBack) (JiData * output, void* userData);

#define MAX_VERSION_LENGTH (512)

#endif // EV_JI_TYPES_H_