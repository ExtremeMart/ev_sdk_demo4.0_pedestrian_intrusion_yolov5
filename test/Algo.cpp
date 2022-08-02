#include "Algo.h"

#include <chrono>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>


const std::string g_db_path = "/usr/local/ev_sdk/bin/face_db";
const int g_lib_id = 0;

struct EvData
{
    int index;
    int inferenceType;
    Algo *obj;
};

Algo::Algo()
{
}

Algo::~Algo()
{
    if(m_thread.joinable())
    {
        LOG(INFO) << "wait thread join ..";
        m_thread.join();
        LOG(INFO) << "exit thread success ..";
    }

    if (m_predictor)
    {
        ji_destroy_predictor(m_predictor);
        m_predictor = nullptr;
        LOG(INFO) << "destroy predictor successfully";
    }
}

bool Algo::Init(const int &type)
{
    m_predictor = ji_create_predictor(JISDK_PREDICTOR_DEFAULT);
    if (m_predictor == nullptr)
    {
        LOG(ERROR) << "create predictor error";
        return -1;
    }

    LOG(INFO) << "sdk mode : " << type;
    m_isAsyn = (type == 0) ? false : true;

    ji_set_callback(m_predictor, AlgoCallback);
}

void Algo::SetOutFileName(const std::string &outFile)
{
    m_outFile = outFile;
    return;
}


bool Algo::SetConfig(const char *args)
{
    JiErrorCode ret = ji_update_config(m_predictor, args);
    if (ret != JISDK_RET_SUCCEED)
    {
        LOG(INFO) << "ji_calc_image error, return " << ret;
        return false;
    }

    return true;
}

bool Algo::ProcessImages(const std::vector<std::string> &filenames, const char *args, int repeate )
{
    JiEvent event;
    JiImageInfo * images = new JiImageInfo[filenames.size()];
    JiImageInfo *outImage = nullptr;
    for(int i = 0; i < filenames.size(); ++i)
    {
    	cv::Mat inMat = cv::imread(filenames[i]);
        if (inMat.empty())
        {
            LOG(ERROR) << "[ERROR] cv::imread source file failed, " << filenames[i];
            delete[] images;
            return JISDK_RET_INVALIDPARAMS;
        }

        images[i].nWidth = inMat.cols;
        images[i].nHeight = inMat.rows;
        images[i].nWidthStride = inMat.cols;
        images[i].nHeightStride = inMat.rows;
        images[i].nFormat = JI_IMAGE_TYPE_BGR;
        images[i].nDataLen = inMat.total();
        images[i].pData = inMat.data;
        images[i].nDataType = JI_UNSIGNED_CHAR;
    }

    int nRepeats = (repeate <= 0) ? 1 : repeate;

    int count = 0;
    while (++count <= nRepeats)
    {
        if (m_isAsyn == true)
        {
            if (m_inferenceSize == 0)
            {
                m_inferenceSize = nRepeats;
                m_runFlag = true;
                std::thread newThread([&]
                                      { run(); });
                m_thread.swap(newThread);
            }

            EvData *userData = new EvData;
            userData->index = count;
            userData->inferenceType = 0;
            userData->obj = this;
            LOG(INFO) << "Call asyn " << count;
            ProcessOneFrameAsyn(images, args, userData);
            continue;
        }

        if (ProcessOneFrame(images, args, &outImage, event))
        {
            LOG(INFO) << "event info:"
                      << "\n\tcode: " << event.code
                      << "\n\tjson: " << event.json;

            if (event.code != JISDK_CODE_FAILED)
            {
                if (outImage != nullptr && !m_outFile.empty())
                {
                    if (outImage[0].nFormat == JI_IMAGE_TYPE_BGR && outImage[0].nDataType == JI_UNSIGNED_CHAR)
                    {
                        cv::Mat outMat(outImage[0].nHeight, outImage[0].nWidth, CV_8UC3, outImage[0].pData);
                        cv::imwrite(m_outFile, outMat);
                        LOG(INFO) << "write out image successfully.";
                    }
                    LOG(INFO) << "get out size : " << m_outSize;
                    for(int i = 1 ; i < m_outSize; ++i)
                    {
                        if (outImage[i].nFormat == JI_IMAGE_TYPE_BGR && outImage[i].nDataType == JI_UNSIGNED_CHAR)
                        {
                            cv::Mat tmpMat = cv::Mat(outImage[i].nHeight, outImage[i].nWidth, CV_8UC3, outImage[i].pData).clone();
                            std::string picName = "/usr/local/ev_sdk/bin/index_" + std::to_string(i) +  ".jpg";
                            cv::imwrite(picName, tmpMat);
                        }
                    }
                }
            }
        }
    }
    delete[] images;
    return true;

}

bool Algo::ProcessImage(const std::string &filename, const char *args, int repeate)
{
    JiEvent event;
    cv::Mat inMat = cv::imread(filename);
    if (inMat.empty())
    {
        LOG(ERROR) << "[ERROR] cv::imread source file failed, " << filename;
        return JISDK_RET_INVALIDPARAMS;
    }

    JiImageInfo image[1];
    JiImageInfo *outImage = nullptr;

    image[0].nWidth = inMat.cols;
    image[0].nHeight = inMat.rows;
    image[0].nWidthStride = inMat.cols;
    image[0].nHeightStride = inMat.rows;
    image[0].nFormat = JI_IMAGE_TYPE_BGR;
    image[0].nDataLen = inMat.total();
    image[0].pData = inMat.data;
    image[0].nDataType = JI_UNSIGNED_CHAR;

    int nRepeats = (repeate <= 0) ? 1 : repeate;

    int count = 0;
    while (++count <= nRepeats)
    {
        if (m_isAsyn == true)
        {
            if (m_inferenceSize == 0)
            {
                m_inferenceSize = nRepeats;
                m_runFlag = true;
                std::thread newThread([&]
                                      { run(); });
                m_thread.swap(newThread);
            }

            EvData *userData = new EvData;
            userData->index = count;
            userData->inferenceType = 0;
            userData->obj = this;
            LOG(INFO) << "Call asyn " << count;
            ProcessOneFrameAsyn(image, args, userData);
            continue;
        }

        if (ProcessOneFrame(image, args, &outImage, event))
        {
            LOG(INFO) << "event info:"
                      << "\n\tcode: " << event.code
                      << "\n\tjson: " << event.json;

            if (event.code != JISDK_CODE_FAILED)
            {
                if (outImage != nullptr && !m_outFile.empty())
                {
                    if (outImage[0].nFormat == JI_IMAGE_TYPE_BGR && outImage[0].nDataType == JI_UNSIGNED_CHAR)
                    {
                        cv::Mat outMat(outImage[0].nHeight, outImage[0].nWidth, CV_8UC3, outImage[0].pData);
                        cv::imwrite(m_outFile, outMat);
                        LOG(INFO) << "write out image successfully.";
                    }
                    LOG(INFO) << "get out size : " << m_outSize;
                    for(int i = 1 ; i < m_outSize; ++i)
                    {
                        if (outImage[i].nFormat == JI_IMAGE_TYPE_BGR && outImage[i].nDataType == JI_UNSIGNED_CHAR)
                        {
                            cv::Mat tmpMat = cv::Mat(outImage[i].nHeight, outImage[i].nWidth, CV_8UC3, outImage[i].pData).clone();
                            std::string picName = "/usr/local/ev_sdk/bin/index_" + std::to_string(i) +  ".jpg";
                            cv::imwrite(picName, tmpMat);
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool Algo::ProcessVideo(const std::string &filename, const char *args, int repeate)
{
    int nRepeats = (repeate <= 0) ? 1 : repeate;
    int count = 0;
    while (++count <= nRepeats)
    {
        LOG(INFO) << "process times " << nRepeats;

        cv::VideoCapture vcapture(filename);
        if (!vcapture.isOpened())
        {
            LOG(ERROR) << "VideoCapture,open video file failed, " << filename;
            break;
        }

        cv::VideoWriter vwriter;

        cv::Mat inMat;
        JiEvent event;
        while (vcapture.read(inMat))
        {
            if (inMat.empty())
            {
                LOG(ERROR) << "cv::imread source file failed, " << filename;
                return false;
            }

            JiImageInfo image[1];
            JiImageInfo *outImage = nullptr;

            image[0].nWidth = inMat.cols;
            image[0].nHeight = inMat.rows;
            image[0].nWidthStride = inMat.cols;
            image[0].nHeightStride = inMat.rows;
            image[0].nFormat = JI_IMAGE_TYPE_BGR;
            image[0].nDataLen = inMat.total();
            image[0].pData = inMat.data;
            image[0].nDataType = JI_UNSIGNED_CHAR;

            if (m_isAsyn == true)
            {
                EvData *userData = new EvData;
                userData->index = count;
                userData->inferenceType = 1;

                ProcessOneFrameAsyn(image, args, userData);
                continue;
            }

            if (ProcessOneFrame(image, args, &outImage, event))
            {
                LOG(INFO) << "event info:"
                          << "\n\tcode: " << event.code
                          << "\n\tjson: " << event.json;

                if (event.code != JISDK_CODE_FAILED)
                {
                    if (outImage[0].nFormat == JI_IMAGE_TYPE_BGR && outImage[0].nDataType == JI_UNSIGNED_CHAR && !m_outFile.empty())
                    {
                        cv::Mat outMat(outImage[0].nHeight, outImage[0].nWidth, CV_8UC3, outImage[0].pData);
                        if (!vwriter.isOpened())
                        {
                            vwriter.open(m_outFile,
                                         cv::VideoWriter::fourcc('X', '2', '6', '4'),
                                         vcapture.get(cv::CAP_PROP_FPS),
                                         outMat.size());
                            if (!vwriter.isOpened())
                            {
                                LOG(ERROR) << "[ERROR] cv::VideoWriter,open video file failed, " << m_outFile;
                                break;
                            }
                        }
                        vwriter.write(outMat);
                        LOG(INFO) << "write out video successfully.";
                    }
                }
            }
        }

        vwriter.release();
        vcapture.release();
    }
}

bool Algo::ProcessStream(const std::string &url)
{
}

bool Algo::ProcessOneFrame(const JiImageInfo *pInFrames, const char *args, JiImageInfo **pOutFrames, JiEvent &event)
{
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int outSize = 0;
    JiErrorCode ret = ji_calc_image(m_predictor, pInFrames, 1, args, pOutFrames, m_outSize, event);
    if (ret != JISDK_RET_SUCCEED)
    {
        LOG(INFO) << "ji_calc_image error, return " << ret;
        return false;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    LOG(INFO) << "Total processing time: " << (double)(duration) / 1000.0 << " ms";

    return true;
}

bool Algo::ProcessOneFrameAsyn(const JiImageInfo *pInFrames, const char *args, void *userData)
{
    ji_calc_image_asyn(m_predictor, pInFrames, 1, args, userData);
}

void Algo::run()
{
    while (m_runFlag)
    {
        if (m_inferenceCount == m_inferenceSize)
        {
            LOG(INFO) << "get all result , total " << m_inferenceCount;
            break;
        }
        usleep(5000);
    }
}

void Algo::IncreaseCount()
{
    ++m_inferenceCount;
}

std::string Algo::GetOutfileName()
{
    return m_outFile;
}

void AlgoCallback(JiData *output, void *userData)
{
    EvData *data = (EvData *)userData;
    Algo *obj = data->obj;
    LOG(INFO) << "get result : " << data->index;
    JiEvent &event = output->event;
    JiImageInfo &outImage = output->outPic;


    LOG(INFO) << "event info:"
              << "\n\tcode: " << event.code
              << "\n\tjson: " << event.json;

    if (data->inferenceType == 0)
    {
        int index = data->index;
        std::string outFile = obj->GetOutfileName();
        if (event.code != JISDK_CODE_FAILED)
        {
            if (!outFile.empty())
            {
                if (outImage.nFormat == JI_IMAGE_TYPE_BGR && outImage.nDataType == JI_UNSIGNED_CHAR && outImage.pData != nullptr)
                {
                    cv::Mat outMat(outImage.nHeight, outImage.nWidth, CV_8UC3, outImage.pData);
                    
                    cv::imwrite(std::to_string(index) + "_" + outFile , outMat);
                    LOG(INFO) << "write out image successfully.";

                    delete [] (unsigned char*)outImage.pData;
                }
            }
        }

        obj->IncreaseCount();
    }

    delete output;
    delete data;
}


bool Algo::FaceInit()
{
/*
    std::string face_db_black = "/tmp/black_list";
    JiErrorCode ret =  ji_face_init(m_predictor,g_db_path.c_str(),face_db_black.c_str());
    LOG(INFO) << "FaceInit return " << ret;
    return true;
*/
}

bool Algo::FaceInsert(const std::string &fileanme)
{
/*  
    JiErrorCode ret = ji_face_insert(m_predictor,fileanme.c_str(),g_lib_id);
    LOG(INFO) << "FaceInsert return " << ret;
    return true;
*/
}

bool Algo::FaceDelete(const std::string &faceId)
{
/*    
    JiErrorCode ret = ji_face_delete(m_predictor,faceId.c_str(),g_lib_id);
    LOG(INFO) << "FaceInsert return " << ret;
    return true;
*/
}

bool Algo::FaceDbExport()
{
/*
    JiErrorCode ret = ji_face_exportDB(m_predictor,g_db_path.c_str(),g_lib_id);
    LOG(INFO) << "FaceDbExport return " << ret;
    return true;   
*/
}
