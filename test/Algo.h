#ifndef _ALGO_TEST_H
#define _ALGO_TEST_H

#include <string>
#include <thread>

#include "ji.h"
#include "ji_utils.h"



void AlgoCallback(JiData * output, void* userData);

class Algo
{

public:
    Algo();
    ~Algo();
    bool Init(const int &type = 0);
    bool SetConfig(const char *args);
    bool ProcessImage(const std::string &filename, const char *args, int repeate = 0);
    bool ProcessImages(const std::vector<std::string> &filenames, const char *args, int repeate = 0);
    bool ProcessVideo(const std::string &filename, const char *args, int repeate = 0);
    bool ProcessStream(const std::string &url);

    bool ProcessOneFrame(const JiImageInfo *pInFrames, const char *args, JiImageInfo **pOutFrames, JiEvent &event);
    bool ProcessOneFrameAsyn(const JiImageInfo *pInFrames, const char *args, void *userData);

    bool FaceInit();
    bool FaceInsert(const std::string &fileanme);
    bool FaceDelete(const std::string &faceId);
    bool FaceDbExport();


    void SetOutFileName(const std::string &outFile);
    void IncreaseCount();

    std::string GetOutfileName();

private:
    void run();
    void setInfSize(const int &size);
    int getInfSize();
    bool m_runFlag = false;
    std::thread m_thread;
    int m_inferenceCount = 0;
    int m_inferenceSize = 0;

    unsigned int m_outSize = 0;

private:
    void *m_predictor = nullptr;
    bool m_isAsyn = false;
    std::string m_outFile;


};

#endif
