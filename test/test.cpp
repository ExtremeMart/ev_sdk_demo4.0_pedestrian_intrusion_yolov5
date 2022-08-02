#include <getopt.h>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "Algo.h"

using namespace std;

#define EMPTY_EQ_NULL(a) ((a).empty()) ? NULL : (a).c_str()

string strFunction, strIn, strArgs, strOut, strUpdateArgs;
bool isThread = false;
int repeats = 1;

enum class CMD{
    //ji_undefie = 0,
    ji_undefine = 0,
    ji_calc_image ,
    ji_calc_image_asyn,
    ji_destroy_predictor,
    ji_thread,
    ji_get_version,
    ji_insert_face,
    ji_delete_face
};


int check_filetype(const string &filename)
{
    int filetype = -1; //0:image; 1:video; 2:folder

    std::size_t found = filename.rfind('.');
    if (found != std::string::npos)
    {
        string strExt = filename.substr(found);        
        std::transform(strExt.begin(), strExt.end(), strExt.begin(), ::tolower);
        if (strExt.compare(".jpg") == 0 ||
            strExt.compare(".jpeg") == 0 ||
            strExt.compare(".png") == 0 ||
            strExt.compare(".bmp") == 0 )
        {
            LOG(INFO) << "file type is " << 0;
            filetype = 0;
        }
        if (strExt.compare(".mp4") == 0 ||
            strExt.compare(".avi") == 0 ||
            strExt.compare(".flv") == 0 ||
            strExt.compare(".mkv") == 0 ||
            strExt.compare(".wmv") == 0 ||
            strExt.compare(".rmvb") == 0)
        {
            LOG(INFO) << "file type is " << 1;
            filetype = 1;
        }        
    }

    struct stat buff;
    int result = stat(filename.c_str(), &buff);
    if( (result == 0) && (buff.st_mode&S_IFMT) == S_IFDIR)
    {
        LOG(INFO) << "file type is " << 2;
        filetype = 2;
    }

    return filetype;
}

void signal_handle(const char *data, int size)
{
    std::ofstream fs("core_dump.log", std::ios::app);
    std::string str = std::string(data, size);

    fs << str;
    LOG(ERROR) << str;

    fs.close();
}

void show_help()
{
    LOG(INFO) << "\n"
              << "---------------------------------\n"
              << "usage:\n"
              << "  -h  --help        show help information\n"
              << "  -f  --function    test function for \n"
              << "                    1.ji_calc_image\n"
              << "                    2.ji_calc_image_asyn\n"
              << "                    3.ji_destroy_predictor\n"
              << "                    4.ji_thread\n"
              << "                    5.ji_get_version\n"
              << "                    6.ji_insert_face\n"
              << "                    7.ji_delete_face\n"
              << "  -i  --infile      source file\n"
              << "  -a  --args        for example roi\n"
              << "  -u  --args        test ji_update_config\n"
              << "  -o  --outfile     result file\n"
              << "  -r  --repeat      number of repetitions. default: 1\n"
              << "                    <= 0 represents an unlimited number of times\n"
              << "                    for example: -r 100\n"
              << "---------------------------------\n";
}

std::vector<std::string> num_pictures(const std::string & aStrIn)
{
     std::string strIn = aStrIn;
     if(strIn[strIn.size()-1]==',')	
     {
         strIn = strIn.substr(0, strIn.size() - 1);
     }
     std::vector<std::string> vecStrParams{};
     auto firstIndex = -1;
     auto secondIndex = 0;
     while(secondIndex != std::string::npos)
     {
         secondIndex = strIn.find(",", firstIndex + 1);
         if(secondIndex != std::string::npos)
         {
             vecStrParams.push_back( strIn.substr(firstIndex + 1, secondIndex - 1 - firstIndex) );
             firstIndex = secondIndex;
         }
         else
         {
             vecStrParams.push_back( strIn.substr(firstIndex + 1, strIn.size()) );
         }
     }     
    return vecStrParams;
}

bool test_for_ji_calc_image()
{
    Algo algoInstance;
    algoInstance.Init();
    algoInstance.SetOutFileName(strOut);
    algoInstance.FaceInit();
    LOG(INFO) << "params----" << strIn;
    int type = check_filetype(strIn);
    if( strUpdateArgs.size() && algoInstance.SetConfig(EMPTY_EQ_NULL(strUpdateArgs)) == false)
    {
        LOG(ERROR) << "ji_update_config error";
        return false;
    }
    
    auto pics = num_pictures(strIn); 
    if( pics.size() > 1 )
    {
        LOG(INFO) << "process multi images:";
        for(const auto& item: pics)
        {
            LOG(INFO) << item;
        }
        algoInstance.ProcessImages(pics, EMPTY_EQ_NULL(strArgs), repeats);
    }
    else if (type == 0)
    {
        algoInstance.ProcessImage(strIn, EMPTY_EQ_NULL(strArgs), repeats);
    }
    else if (type == 1)
    {
        algoInstance.ProcessVideo(strIn, EMPTY_EQ_NULL(strArgs), repeats);
    }
    else if (type == 2)
    {
        DIR *pDir;
        struct dirent* ptr;
        if( !(pDir = opendir(strIn.c_str())) )
        {
            LOG(ERROR) << "path not exists : " << strIn;
            return false;
        }      
        std::vector<std::string>  vecImgNames{}; 
        while( (ptr = readdir(pDir))!=0 )
        {
            if(check_filetype(ptr->d_name) == 0)
            {
                std::string filename = strIn;
                if( filename[filename.size()-1] != '/')
                {
                    filename = filename +'/';
                }
                filename = filename + ptr->d_name;
                vecImgNames.push_back(filename);                
            }
        }
        for(auto &filename: vecImgNames)        
        {
            LOG(INFO) << "process image : " << filename;
            size_t sep = filename.rfind('.');
            std::string outname = filename.substr(0,sep) + "_result" + filename.substr(sep);
            algoInstance.SetOutFileName(outname);
            algoInstance.ProcessImage(filename, EMPTY_EQ_NULL(strArgs), repeats);
        }
    }
    return true;
}

bool test_for_ji_calc_image_asyn()
{
    Algo algoInstance;
    algoInstance.Init(1);
    algoInstance.SetOutFileName(strOut);
    int type = check_filetype(strIn);

    if(algoInstance.SetConfig(EMPTY_EQ_NULL(strUpdateArgs)) == false)
    {
        LOG(ERROR) << "ji_update_config error";
        return false;
    }

    if (type == 0)
    {
        algoInstance.ProcessImage(strIn, EMPTY_EQ_NULL(strArgs), repeats);
    }
    else
    {
        LOG(INFO) << "Not implemented";
    }

    return true;
}

void test_for_ji_destroy_predictor()
{
    int count = 0;
    int nRepeats = (repeats <= 0) ? 1 : repeats;
    while (++count < nRepeats)
    {
        test_for_ji_calc_image();
    }
}

void *threadExec(void *p)
{
    int num = *(int *)p;
    Algo algoInstance;
    algoInstance.Init();

    if(!strOut.empty())
    {
        string fileName = "thread_" + to_string(num) + "_" + strOut;
        algoInstance.SetOutFileName(fileName);
    }
    
    int type = check_filetype(strIn);

    do
    {
        if(algoInstance.SetConfig(EMPTY_EQ_NULL(strUpdateArgs)) == false)
        {
            LOG(ERROR) << "ji_update_config error";
            break;
        }

        if (type == 0)
        {
            algoInstance.ProcessImage(strIn, EMPTY_EQ_NULL(strArgs), repeats);
        }
        else
        {
            algoInstance.ProcessVideo(strIn, EMPTY_EQ_NULL(strArgs), repeats);
        }
    } while (0);
    
}

void test_for_thread()
{
    int threadNum = 5;
    pthread_t thread[5];
    int num[5] = {0, 1, 2, 3, 4};

    for (int i = 0; i < threadNum; i++)
    {
        pthread_create(&thread[i], NULL, threadExec, &num[i]);
    }

    for (int i = 0; i < threadNum; i++)
    {
        LOG(INFO) << "wait thread " << i + 1 << " stop ..";
        pthread_join(thread[i], NULL);
    }
}

void test_for_ji_get_version()
{
    char versionInfo[1024] = {0};
    JiErrorCode ret =  ji_get_version(versionInfo);
    LOG(INFO) << "ji_get_version return " << ret;
    LOG(INFO) << versionInfo;
}

void test_ji_insert_face()
{
    LOG(INFO) << "start ji_insert_face";
    Algo algoInstance;
    algoInstance.Init();
    algoInstance.SetOutFileName(strOut);
    algoInstance.FaceInit();
    int type = check_filetype(strIn);
    if(type == 0)
    {
        algoInstance.FaceInsert(strIn);
        algoInstance.FaceDbExport();
    }

    return;
}

void test_ji_delete_face()
{
    LOG(INFO) << "start ji_insert_face";
    Algo algoInstance;
    algoInstance.Init();
    algoInstance.SetOutFileName(strOut);
    algoInstance.FaceInit();
    int type = check_filetype(strIn);
    if(type == 0)
    {
        string faceId = strIn;
        algoInstance.FaceDelete(strIn);
        algoInstance.FaceDbExport();
    }

    return;
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_minloglevel = google::INFO;
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;

    /* 如需要保存日志到文件test.INFO,请启用下一行代码 */
    //google::SetLogDestination(google::GLOG_INFO, "info");

    //capture exception
    google::InstallFailureSignalHandler();
    google::InstallFailureWriter(&signal_handle);

    //parse params
    const char *short_options = "hf:l:i:a:o:r:u:";
    const struct option long_options[] = {
        {"help", 0, NULL, 'h'},
        {"function", 1, NULL, 'f'},
        {"infile", 1, NULL, 'i'},
        {"args", 1, NULL, 'a'},
        {"outfile", 1, NULL, 'o'},
        {"repeat", 1, NULL, 'r'},
        {"update", 1, NULL, 'u'},
        {0, 0, 0, 0}};

    bool bShowHelp = false;
    int c;
    int option_index = 0;
    do
    {
        c = getopt_long(argc, argv, short_options, long_options, &option_index);
        if (c == -1)
            break;

        switch (c)
        {
        case 'h':
            bShowHelp = true;
            break;

        case 'f':
            strFunction = optarg;
            break;

        case 'i':
            strIn = optarg;
            break;

        case 'a':
            strArgs = optarg;
            break;

        case 'o':
            strOut = optarg;
            break;

        case 'r':
            repeats = atoi(optarg);
            break;
        
        case 'u':
            strUpdateArgs = optarg;
            break;

        default:
            break;
        }
    } while (1);

    if (bShowHelp)
    {
        show_help();
        return 0;
    }

    //check params
    c = -1;
    CMD command = CMD::ji_undefine;
    if (strFunction.compare("ji_calc_image") == 0 || strFunction.compare(to_string(static_cast<int>(CMD::ji_calc_image))) == 0)
        command = CMD::ji_calc_image;
    else if (strFunction.compare("ji_calc_image_asyn") == 0 || strFunction.compare(to_string(static_cast<int>(CMD::ji_calc_image_asyn))) == 0)
        command = CMD::ji_calc_image_asyn;
    else if (strFunction.compare("ji_destroy_predictor") == 0 || strFunction.compare(to_string(static_cast<int>(CMD::ji_destroy_predictor))) == 0)
        command = CMD::ji_destroy_predictor;
    else if (strFunction.compare("ji_thread") == 0 || strFunction.compare(to_string(static_cast<int>(CMD::ji_thread))) == 0)
        command = CMD::ji_thread;
    else if (strFunction.compare("ji_get_version") == 0 || strFunction.compare(to_string(static_cast<int>(CMD::ji_get_version))) == 0)
        command = CMD::ji_get_version;
    else if (strFunction.compare("ji_insert_face") == 0 || strFunction.compare(to_string(static_cast<int>(CMD::ji_insert_face))) == 0)
        command = CMD::ji_insert_face;
    else if (strFunction.compare("ji_delete_face") == 0 || strFunction.compare(to_string(static_cast<int>(CMD::ji_delete_face))) == 0)
        command = CMD::ji_delete_face;

    if (command == CMD::ji_undefine)
    {
        LOG(ERROR) << "[ERROR] invalid function.";
        show_help();
        return -1;
    }

    if (command != CMD::ji_get_version && strIn.empty())
    {
        LOG(ERROR) << "[ERROR] no infile.";
        show_help();
        return -1;
    }

    if (repeats <= 0) 
    {
        repeats = -1;
    }

    //print params
    static const string cs_function[] = {
        "1.ji_calc_image",
        "2.ji_calc_image_asyn",
        "3.ji_destroy_predictor",
        "4.ji_thread",
        "5.ji_get_version"};

    LOG(INFO) << "run params info:"
              << "\n\tfuction: " << cs_function[static_cast<int>(command) - 1]
              << "\n\tinfile: " << strIn
              << "\n\targs: " << strArgs
              << "\n\toutfile: " << strOut
              << "\n\trepeat:" << repeats;
    ji_init(argc,argv);
    switch (command)
    {
        case CMD::ji_calc_image:
        {
            test_for_ji_calc_image();
            break;
        }   
        case CMD::ji_calc_image_asyn:
        {
            test_for_ji_calc_image_asyn();
            break;
        }
        case CMD::ji_destroy_predictor:
        {
            test_for_ji_destroy_predictor();
            break;
        }
        case CMD::ji_thread:
        {
            isThread = true;
            test_for_thread();
            break;
        }
        case CMD::ji_get_version:
        {
            test_for_ji_get_version();
            break;
        }
        case CMD::ji_insert_face:
        {
            test_ji_insert_face();
            break;
        }
        case CMD::ji_delete_face:
        {
            test_ji_delete_face();
            break;
        }
        default:
            break;
    }
    ji_reinit();
    return 0;
}
