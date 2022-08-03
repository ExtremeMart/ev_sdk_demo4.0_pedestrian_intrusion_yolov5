# EV_SDK

## 说明
### EV_SDK的目标
开发者专注于算法开发及优化，最小化业务层编码，即可快速部署到生产环境，共同打造商用级高质量算法。
### 极市平台做了哪些
1. 统一定义算法接口：针对万千视频和图片分析算法，抽象出接口，定义在`include`目录下的`ji.h`文件中
2. 提供工具包：比如jsoncpp库，wkt库，在`3rd`目录下
3. 应用层服务：此模块不在ev_sdk中，比如视频处理服务、算法对外通讯的http服务等

### 开发者需要做什么
1. 模型的训练和调优
2. 实现`ji.h`约定的接口
3. 实现约定的输入输出
4. 其他后面文档提到的功能

## 目录

### 代码目录结构

```
ev_sdk
|-- 3rd             # 第三方源码或库目录，发布时请删除
|   |-- wkt_parser          # 针对使用WKT格式编写的字符串的解析器
|   |-- jsoncpp_simple         # jsoncpp库，简单易用
|   `-- fonts         # 支持中文画图的字体库
|-- CMakeLists.txt          # 本项目的cmake构建文件
|-- README.md       # 本说明文件
|-- model           # 模型数据存放文件夹
|-- config          # 程序配置目录
|   |-- README.md   # algo_config.json文件各个参数的说明和配置方法
|   `-- algo_config.json    # 程序配置文件
|-- doc
|-- include         # 库头文件目录
|   `-- ji.h        # libji.so的头文件，理论上仅有唯一一个头文件
|-- lib             # 本项目编译并安装之后，默认会将依赖的库放在该目录，包括libji.so
|-- src             # 实现ji.cpp的代码
`-- test            # 针对ji.h中所定义接口的测试代码，请勿修改！！！
```

## <span id="jump">!!开发SDK的一般步骤(开发者重点关注的部分)</span>

1. 获取demo源码,将源码拷贝至目标路径/usr/local/ev_sdk（**注意！！！整个demo是属于一个git工程，避免将git工程的隐藏文件，如.git目录，放到目标路径/usr/local/ev_sdk中,因为ev_sdk目录也是一个git工程目录；建议是拷贝而不是直接把demo的目录名称重命名为ev_sdk而把原始ev_sdk覆盖掉或者把原始ev_sdk工程目录删掉；否则容易引起一些未知的错误**）,下载对应的模型到/usr/local/ev_sdk/model目录下并修改模型文件名(**本demo的git仓库自带模型，无需下载**).
   
```
   !!!在使用yolov5官方源码导出onnx模型时只支持tensort主版本为8的推理
   若tensorrt主版本为7时需要修改导出代码，导出代码见3rd/export.py               
```

2. 添加修改算法业务逻辑代码后编译ev_sdk，注意修改ji.cpp中的版本号algo_version,执行完成之后，`/usr/local/ev_sdk/lib`下将生成`libji.so`和相关的依赖库，以及`/usr/local/ev_sdk/bin/`下的测试程序`test-ji-api`。需要注意的是**一定要有install,目的是将相应地库都安装到ev_sdk/lib下面**
  ```shell
        #编译SDK库
        mkdir -p /usr/local/ev_sdk/build
        cd /usr/local/ev_sdk/build
        cmake ..
        make install 
        #编译测试工具
        mkdir -p /usr/local/ev_sdk/test/build
        cd /usr/local/ev_sdk/test/build
        cmake ..
        make install 
  ```
3. 测试,我们提供了测试工具test-ji-api方便开发者进行算法的快速测试,在下面的[demo测试](#jump_dev_test)中我们将演示如何使用测试工具进行基本的测试(**这点非常重要,开发者需要先自测通过再提交算法**)

## demo说明
本项目基于极视平台EV_SDK4.0标准作为模板进行算法封装，旨在为开发者提供最直观的关于SDK4.0的理解. 本项目中我们利用**YOLOV5目标检测框架,并采用Tensorrt进行模型推理**, 实现一个行人检测算法, 算法逻辑为当行人出现在指定ROI内部时触发报警,在算法返回图片及json中进行体现.本项目中算法首次运行时,先用Tensorrt直接加载onnx模型,并保存为trt模型,以便再次运行时直接trt模型,加快初始化.本项目中,我们使用了一般SDK开发中都会用的的配置解析,json字符串处理, 图片文字(框)绘制等功能,开发者可以参考demo中的部分逻辑.
### demo源码介绍
我们在demo开发中采用单一职责原则,将算法逻辑处理,模型推理,配置管理等功能封装到不同的C++类中,方便算法的开发和维护.
1. 在SampleAlgorithm.cpp文件中我们封装了算法逻辑,通过该对象的实现算法的初始化,调用,反初始化,配置更新等功能.对象内部保存json字符串的成员变量,以及返回图片对象的成员变量,如果开发这按照如下的几个接口封装算法类,则只需在ji.cpp中更改算法版本信息algo_version,无需做其他更改.

    ```
    在ji.cpp ji_predictor_detector 中调用算法的初始化接口:        
        STATUS Init();

    在ji.cpp ji_calc_image 中调用算法分析接口和返回图片获取接口:
        STATUS Process(const Mat &inFrame, const char *args, JiEvent &event);
        STATUS GetOutFrame(JiImageInfo **out, unsigned int &outCount);
    
    在ji.cpp ji_destroy_detector 中调用算法反初始化接口:
        STATUS UnInit();

    在ji.cpp ji_update_config 中调用算法配置更新接口:
        STATUS UpdateConfig(const char *args);
    ```
2. 在Configuration.hpp文件中,我们封装了一个简单易用的配置管理工具,其中已经包含了一些常见的配置项,开发者需要根据算法需求自行增加或者删减配置项,下面演示如何基于配置管理类快速增加配置.假设我们需要增加一个nms阈值,只需在Configuration.hpp中增加如下两句,删除配置像亦然.
```
    struct Configuration
    {
        ..............
        float nms_thresh = 0.45; //增加成员
        ..............
        void ParseAndUpdateArgs(const char *confStr)
        {
            ..............
            checkAndUpdateNumber("nms_thresh", nms_thresh); //增加更新操作
            ..............
        }
        ..............
    }    
```
3. SampleDetector.cpp 中我们封装了一个基于tensorrt的YoloV5目标检测类,用来检测行人.
4. demo运行的基础镜像及容器启动命令
   
```
运行镜像环境
ccr.ccs.tencentyun.com/public_images/ubuntu16.04-cuda11.1-cudnn8.0-ffmpeg4.2-opencv4.1.2-tensorrt7.2.3-code-server3.5.0-ev_base:v1.0

容器运行命令
nvidia-docker run -itd --privileged ccr.ccs.tencentyun.com/public_images/ubuntu16.04-cuda11.1-cudnn8.0-ffmpeg4.2-opencv4.1.2-tensorrt7.2.3-code-server3.5.0-ev_base:v1.0 /bin/bash
```

### [demo测试](#jump_dev_test)
我们按照[开发SDK的一般步骤](#jump_dev)编译并授权过后即可运行测试工具,测试工具主要提供一下几个功能

```
        ---------------------------------
          usage:
            -h  --help        show help information
            -f  --function    test function for 
                              1.ji_calc_image
                              2.ji_calc_image_asyn
                              3.ji_destroy_predictor
                              4.ji_thread
                              5.ji_get_version
                              6.ji_insert_face
                              7.ji_delete_face
            -i  --infile      source file
            -a  --args        for example roi
            -u  --args        test ji_update_config
            -o  --outfile     result file
            -r  --repeat      number of repetitions. default: 1
                              <= 0 represents an unlimited number of times
                              for example: -r 100
        ---------------------------------

```

下面我们对部分功能进行详细的说明(未说明的参数暂未实现)

1. 指定调用功能的-f参数
    1. -f 1指调用算法同步分析接口，调用该接口时主要支持如下几种输入方式:

    ```"shell"
        1.输入单张图片，需要指定输入输出文件
        　　./test-ji-api -f 1 -i ../data/persons.jpg -o result.jpg
        
        2.输入多张图片(多张图片是指sdk一次调用ji_calc_image传入的图片数量不是指多次调用，每次传入一张图片)，需要指定输入输出文件
            　./test-ji-api -f 1 -i ../data/persons.jpg,../data/persons.jpg -o result.jpg #输入两张图片
        
        3.输入视频，需要指定输入输出文件
        　　./test-ji-api -f 1 -i ../data/test.mp4 -o test_result.mp4 

        4.输入图片文件夹，只需指定输入文件夹即可，结果图片会保存在原图片文件的同一路径下，结果文件名和原文件名一一对应(名称中添加了result字段)
        　　　./test-ji-api -f 1 -i ../data/　
             图片列表文件格式如下:   
                /usr/local/ev_sdk/data/a.jpg
                /usr/local/ev_sdk/data/b.jpg
                /usr/local/ev_sdk/data/c.png   
    ```

    2. -f 3指调用算法实例创建释放接口，该接口需要与-r参数配合使用，测试在循环创建/调用/释放的过程中是否存在内存/显存的泄露，与调用-f 1的区别在于，当-r参数指定调用次数时，-f 1只会创建一次实例，释放一次实例．

    ```'shell'
        ./test-ji-api -f 3 -i ../data/persons.jpg -o result.jpg -r -1 #无限循环调用

        ./test-ji-api -f 3 -i ../data/persons.jpg -o result.jpg -r 100 #循环调用100次
    ```

    3. -f 5获取并打印算法的版本信息．

    ```'shell'
        ./test-ji-api -f 5
    ```
2. 指定输入的-i参数，使用方式见上文介绍.
3. 指定输出的-o参数，使用方式见上文介绍. 
4. 指定配置的-u/-a参数,算法初始化时会从配置文件中加载默认配置参数,对于部分参数通过接口可以动态覆盖默认参数,如果项目要求能够动态指定的参数,需要测试通过-u和-a传递的参数能够生效.例如,对于本demo的配置文件如下

    ```"json"
   {
    "draw_roi_area": true,
    
    "roi_type": "polygon_1",
    "polygon_1": ["POLYGON((0 0, 1 0, 1 1, 0 1))"],

    "roi_color": [255, 255, 0, 0.7],
    "roi_line_thickness": 4,
    "roi_fill": false,
    "draw_result": true,
    "draw_confidence": true,
    "thresh": 0.1,
    "language": "en",

    "target_rect_color": [0, 0, 255, 0],
    "object_rect_line_thickness": 3,
    "object_text_color": [255, 255, 255, 0],
    "object_text_bg_color": [50, 50, 50, 0],
    "object_text_size": 30,
    "mark_text_en": ["person"],
    "mark_text_zh": ["人体"],
    "draw_warning_text": true,
    "warning_text_en": "WARNING! WARNING!",
    "warning_text_zh": "警告!",
    "warning_text_size": 30,
    "warning_text_color": [255, 255, 255, 0],
    "warning_text_bg_color": [0, 0, 200, 0],
    "warning_text_left_top": [0, 0]
    }
    ```

配置文件中的polygon_1参数和language参数需要支持动态配置,则需要利用-a和-u参数测试,-u和-a参数的区别在于-u是通过ji_update_config接口单独传递,-i是通过ji_calc_iamge的args参数传递.
   
   ```"shell"
     //-u指定参数
        ./test-ji-api -f 1 -i ../data/persons.jpg 
        -u "{\"polygon_1\": [\"POLYGON((0.2 0.2, 0.8 0, 0.8 0.8, 0 0.8))\"],\"language\":\"zh\"}"
        -o result.jpg

    //-a指定参数
        ./test-ji-api -f 1 -i ../data/persons.jpg  
        -a "{\"polygon_1\": [\"POLYGON((0.2 0.2, 0.8 0, 0.8 0.8, 0 0.8))\"],\"language\":\"zh\"}"
        -o result.jpg
   ```

以下为默认参数的输出效果  

![alt 默认参数](doc/figure1.jpg)

```
    code: 0
        json: {
        "algorithm_data" : 
        {
                "is_alert" : true,
                "target_info" : 
                [
                        {
                                "confidence" : 0.81,
                                "height" : 245,
                                "name" : "person",
                                "width" : 86,
                                "x" : 66,
                                "y" : 92
                        },
                        {
                                "confidence" : 0.87,
                                "height" : 229,
                                "name" : "person",
                                "width" : 89,
                                "x" : 260,
                                "y" : 104
                        },
                        {
                                "confidence" : 0.91,
                                "height" : 239,
                                "name" : "person",
                                "width" : 89,
                                "x" : 348,
                                "y" : 112
                        },
                        {
                                "confidence" : 0.92,
                                "height" : 235,
                                "name" : "person",
                                "width" : 77,
                                "x" : 155,
                                "y" : 123
                        },
                        {
                                "confidence" : 0.92,
                                "height" : 256,
                                "name" : "person",
                                "width" : 69,
                                "x" : 0,
                                "y" : 81
                        },
                        {
                                "confidence" : 0.94,
                                "height" : 281,
                                "name" : "person",
                                "width" : 110,
                                "x" : 488,
                                "y" : 98
                        }
                ]
        },
        "model_data" : 
        {
                "objects" : 
                [
                        {
                                "confidence" : 0.81,
                                "height" : 245,
                                "name" : "person",
                                "width" : 86,
                                "x" : 66,
                                "y" : 92
                        },
                        {
                                "confidence" : 0.87,
                                "height" : 229,
                                "name" : "person",
                                "width" : 89,
                                "x" : 260,
                                "y" : 104
                        },
                        {
                                "confidence" : 0.91,
                                "height" : 239,
                                "name" : "person",
                                "width" : 89,
                                "x" : 348,
                                "y" : 112
                        },
                        {
                                "confidence" : 0.92,
                                "height" : 235,
                                "name" : "person",
                                "width" : 77,
                                "x" : 155,
                                "y" : 123
                        },
                        {
                                "confidence" : 0.92,
                                "height" : 256,
                                "name" : "person",
                                "width" : 69,
                                "x" : 0,
                                "y" : 81
                        },
                        {
                                "confidence" : 0.94,
                                "height" : 281,
                                "name" : "person",
                                "width" : 110,
                                "x" : 488,
                                "y" : 98
                        }
                ]
        }
    }
```

以下为指定参数的输出效果  

![alt 默认参数](doc/figure2.jpg)

```"json"
 code: 0
        json: {
        "algorithm_data" : 
        {
                "is_alert" : true,
                "target_info" : 
                [
                        {
                                "confidence" : 0.81,
                                "height" : 245,
                                "name" : "人体",
                                "width" : 86,
                                "x" : 66,
                                "y" : 92
                        },
                        {
                                "confidence" : 0.87,
                                "height" : 229,
                                "name" : "人体",
                                "width" : 89,
                                "x" : 260,
                                "y" : 104
                        },
                        {
                                "confidence" : 0.91,
                                "height" : 239,
                                "name" : "人体",
                                "width" : 89,
                                "x" : 348,
                                "y" : 112
                        },
                        {
                                "confidence" : 0.92,
                                "height" : 235,
                                "name" : "人体",
                                "width" : 77,
                                "x" : 155,
                                "y" : 123
                        }
                ]
        },
        "model_data" : 
        {
                "objects" : 
                [
                        {
                                "confidence" : 0.81,
                                "height" : 245,
                                "name" : "人体",
                                "width" : 86,
                                "x" : 66,
                                "y" : 92
                        },
                        {
                                "confidence" : 0.87,
                                "height" : 229,
                                "name" : "人体",
                                "width" : 89,
                                "x" : 260,
                                "y" : 104
                        },
                        {
                                "confidence" : 0.91,
                                "height" : 239,
                                "name" : "人体",
                                "width" : 89,
                                "x" : 348,
                                "y" : 112
                        },
                        {
                                "confidence" : 0.92,
                                "height" : 235,
                                "name" : "人体",
                                "width" : 77,
                                "x" : 155,
                                "y" : 123
                        },
                        {
                                "confidence" : 0.92,
                                "height" : 256,
                                "name" : "人体",
                                "width" : 69,
                                "x" : 0,
                                "y" : 81
                        },
                        {
                                "confidence" : 0.94,
                                "height" : 281,
                                "name" : "人体",
                                "width" : 110,
                                "x" : 488,
                                "y" : 98
                        }
                ]
        }
    }
```

#### 规范要求
规范测试大部分内容依赖于内置的`/usr/local/ev_sdk/test`下面的代码，这个测试程序会链接`/usr/local/ev_sdk/lib/libji.so`库，`EV_SDK`封装完成提交后，极市方会使用`test-ji-api`程序测试`ji.h`中的所有接口。测试程序与`EV_SDK`的实现没有关系，所以请**请不要修改`/usr/local/ev_sdk/test`目录下的代码！！！**

1. 接口功能要求
  
   - 确定`test-ji-api`能够正常编译，并且将`test-ji-api`移动到任意目录，都需要能够正常运行；
   
   - 在提交算法之前，请自行通过`/usr/local/ev_sdk/bin/test-ji-api`测试接口功能是否正常；
   
   - 未实现的接口需要返回`JISDK_RET_UNUSED`；
   
   - 实现的接口，如果传入参数异常时，需要返回`JISDK_RET_INVALIDPARAMS`；
   
   - 输入图片和输出图片的尺寸应保持一致；
   
   - 对于接口中传入的参数`args`，根据项目需求，算法实现需要支持`args`实际传入的参数。
   
     例如，如果项目需要支持在`args`中传入`roi`参数，使得算法只对`roi`区域进行分析，那么**算法内部必须实现只针对`roi`区域进行分析的功能**；
   
   - 通常输出图片中需要画`roi`区域、目标框等，请确保这一功能正常，包括但不仅限于：
   
     - `args`中输入的`roi`需要支持多边形
     - 算法默认分析区域必须是全尺寸图，如当`roi`传入为空时，算法对整张图进行分析；
     
   - 为了保证多个算法显示效果的一致性，与画框相关的功能必须优先使用`ji_utils.h`中提供的工具函数；
   
   > 1. ` test-ji-api`的使用方法可以参考上面的使用示例以及运行`test-ji-api --help`；
   > 2. 以上要求在示例程序`ji.cpp`中有实现；
   
2. 业务逻辑要求

   针对需要报警的需求，算法必须按照以下规范输出结果：
   * 报警时输出：`JI_EVENT.code=JISDK_CODE_ALARM`，`JI_EVENT.json`内部填充`"is_alert" : true`；
   * 未报警时输出：`JI_EVENT.code=JISDK_CODE_NORMAL`，`JI_EVENT.json`内部填充`"is_alert" : false`；
   * 处理失败的接口返回`JI_EVENT.code=JISDK_CODE_FAILED`


3. 算法配置选项要求

   - **配置文件必须遵循[`EV_SDK`配置协议](./doc/EV_SDK配置协议说明.md)**；
   - 所有算法与`SDK`可配置参数**必须**存放在统一的配置文件：`/usr/local/ev_sdk/config/algo_config.json`中；
   - 配置文件中必须实现的参数项：
     - `draw_roi_area`：`true`或者`false`，是否在输出图中绘制`roi`分析区域；
     - `roi_line_thickness`：ROI区域的边框粗细；
     - `roi_fill`：是否使用颜色填充ROI区域；
     - `roi_color`：`roi`框的颜色，以BGRA表示的数组，如`[0, 255, 0, 0]`，参考[model/README.md](model/README.md)；
     - `roi`：针对图片的感兴趣区域进行分析，如果没有此参数或者此参数解析错误，则roi默认值为整张图片区域；
       注：多个点、线、框有两种实现方式：

       - 使用WKT格式，如：`"roi": "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (30 20, 20 15, 20 25, 30 20)))"`；
       - 使用数组形式，如：`"roi":["POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))", "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"]`。

       **`config/README.md`内必须说明使用的是哪一种格式。**

     - `thresh`：算法阈值，需要有可以调整算法灵敏度、召回率、精确率的阈值参数，如果算法配置项有多个参数，请自行扩展，所有与算法效果相关并且可以变动的参数**必须**在`/usr/local/ev_sdk/config/README.md`中提供详细的配置方法和说明（包括类型、取值范围、建议值、默认值、对算法效果的影响等）；
     - `draw_result`：`true`或者`false`，是否绘制分析结果，比如示例程序中，如果检测到狗，是否将检测框和文字画在输出图中；
     - `draw_confidence`：`true`或者`false`，是否将置信度画在检测框顶部，小数点后保留两位；
     - `language`：所显示文字的语言，需要支持`en`和`zh`两种选项，分别对应英文和中文；
     - 所有`json`内的键名称必须是小写字母，并且单词间以下划线分隔，如上面几个示例。
   - **必须支持参数实时更新**。所有`/usr/local/ev_sdk/config/algo_config.json`内的可配置参数必须支持能够在调用`ji_calc_image`、`ji_calc_image_asyn`接口时，进行实时更新。也就是必须要在`ji_calc_*`等接口的`args`参数中，加入这些可配置项。

     根据算法的实际功能和使用场景，参数实时更新功能可能只能够使部分参数有效，其中

     1. 可以通过`ji_calc_image`等接口的`args`参数传入并实时更新的参数，比如示例代码中检测框的颜色`target_rect_color`，这些配置项称为**动态参数**（即可动态变更）；

     2. 其他无法通过`args`参数传入并进行实时更新的参数称为**静态参数**，通常这些参数需要重启算法实例才能生效；

        > **静态参数的名称规范**：
        >
        > 静态参数必须以`static`作为前缀，例如`static_detect_thresh`。

   - **算法开发完成后，必须按照`config/README.md`的模版，修改成当前算法的配置说明**。

4. 算法输出规范要求

   - **算法输出必须遵循[极视算法SDK输出协议](./doc/极视算法SDK输出协议.md)**; 
   - 算法必须要输出基础模型的结果，否则**不予通过**测试
   - `model_data`表示模型输出的原始数据，用于评估原始算法模型的精度,该键值中的`name`或`class`字段一定要与groundtruth中的完全相同，否则会影响模型性能评估

5. 文件结构规范要求

   * 与模型相关的文件必须存放在`/usr/local/ev_sdk/model`目录下，例如权重文件、目标检测通常需要的名称文件`coco.names`等。
   * 最终编译生成的`libji.so`必须自行链接必要的库，`test-ji-api`不会链接除`/usr/local/ev_sdk/lib/libji.so`以外的算法依赖库；
   * 如果`libji.so`依赖了系统动态库搜索路径（如`/usr/lib`，`/lib`等）以外的库，必须将其安装到`/usr/local/ev_sdk/lib`下，可以使用`ldd /usr/local/ev_sdk/lib/libji.so`查看`libji.so`是否正确链接了所有的依赖库。

## FAQ

### 如何使用接口中的`args`？

通常，在实际项目中，外部需要将多种参数（例如`ROI`）传入到算法，使得算法可以根据这些参数来改变处理逻辑。`EV_SDK`接口（如`int ji_calc_image(void* predictor, const JiImageInfo* pInFrames, const unsigned int nInCount, const char* args,JiImageInfo **pOutFrames, unsigned int & nOutCount, JiEvent &event)`中的`args`参数通常由开发者自行定义和解析，但只能使用[JSON](https://www.json.cn/wiki.html)格式。格式样例：

```shell
{
    "polygon_1": [
        "POLYGON((0.0480.357,0.1660.0725,0.3930.0075,0.3920.202,0.2420.375))",
        "POLYGON((0.5130.232,0.790.1075,0.9280.102,0.9530.64,0.7590.89,0.510.245))",
        "POLYGON((0.1150.497,0.5920.82,0.5810.917,0.140.932))"
    ]
}
```

例如当算法支持输入`polygon_1`参数时，那么开发者需要在`EV_SDK`的接口实现中解析上面示例中`polygon_1`这一值，提取其中的`polygon_1`参数，并使用`WKTParser`对其进行解析，应用到自己的算法逻辑中。

### 为什么要定义roi_type字段
不同算法需要的点线框格式不同，为了保证上层应用能正确地下发`args`参数，需要开发者通过`roi_type`字段说明算法支持的类型，如：
```shell
{
   "roi_type":"polygon_1;"
   "polygon_1": ["POLYGON((0.0480.357,0.1660.0725,0.3930.0075,0.3920.202,0.2420.375))"]
}
```

### 为什么不能且不需要修改`/usr/local/ev_sdk/test`下的代码？

1. `/usr/local/ev_sdk/test`下的代码是用于测试`ji.h`接口在`libji.so`中是否被正确实现，这一测试程序与`EV_SDK`的实现无关，且是极市方的测试标准，不能变动；
2. 编译后`test-ji-api`程序只会依赖`libji.so`，如果`test-ji-api`无法正常运行，很可能是`libji.so`没有按照规范进行封装；

### 为什么运行`test-ji-api`时，会提示找不到链接库？

由于`test-ji-api`对于算法而言，只链接了`/usr/local/ev_sdk/lib/libji.so`库，如果`test-ji-api`运行过程中，找不到某些库，那么很可能是`libji.so`依赖的某些库找不到了。此时

1. 可以使用`ldd /usr/local/ev_sdk/lib/libji.so`检查是否所有链接库都可以找到；
2. 请按照规范将系统动态库搜索路径以外的库放在`/usr/local/ev_sdk/lib`目录下。

### 如何使用`test-ji-api`进行测试？

1. 输入单张图片，并调用`ji_calc_image`接口：

   ```shell
   ./test-ji-api -f 1 -i /path/to/test.jpg 
   ```

2. 输入`json`格式的`polygon_1`参数到`args`参数：

   ```shell
   ./test-ji-api \
   -f 1 \   
   -i /path/to/test.jpg \
   -a '{"polygon_1":["POLYGON((0.2 0.2,0.7 0.13,0.9 0.7,0.4 0.9,0.05 0.8,0.2 0.25))"]}'
   ```

3. 保存输出图片：

   ```shell
   ./test-ji-api -f 1 -i /path/to/test.jpg -o /path/to/out.jpg
   ```

更多选项，请参考`test-ji-api --help`

### EV_SDK配置协议的实现样例

[配置协议](doc/EV_SDK配置协议说明.md)规定：

1. 只能有一级*KEY-VALUE*；
2. *VALUE*的类型有两种：
   1. JSON格式定义的非数组和非对象类型，如`string`、`number`、`false`、`true`；
   2. 由配置协议所定义的数据类型；

举例：

1. 算法支持多个ROI时的配置，*VALUE*可以使用协议所定义的多个`POLYGON`类型：

   ```json
   {
     "roi_type":"polygon_1;",
     "polygon_1": ["POLYGON ((0.1 0.1, 0.1 0.1, 0.2 0.3, 0.9 0.9))", "POLYGON ((0.1 0.1, 0.1 0.1, 0.2 0.3, 0.9 0.9))"]
   }
   ```

2. 算法需要多个ROI，并且多个每个ROI表示不同逻辑含义时：

   ```json
   {
     "roi_type":"polygon_1;polygon_2",
     "polygon_1": ["POLYGON ((0.1 0.1, 0.1 0.1, 0.2 0.3, 0.9 0.9))"],
     "polygon_2": ["POLYGON ((0.1 0.1, 0.1 0.1, 0.2 0.3, 0.9 0.9))"]
   }
   ```

   算法内部根据`polygon_1`和`polygon_2`进行逻辑区分。

3. 如果算法需要以组合的形式配置算法，且组合的数量不设限制时，可用如下配置：

   ```json
   {

     "polygon_1": ["POLYGON ((0.1 0.1, 0.1 0.1, 0.2 0.3, 0.9 0.9))"],
     "line_1": ["LINESTRING (0.1 0.1, 0.12 0.15, 0.2 0.3)"],
     "polygon_2": ["POLYGON ((0.1 0.1, 0.1 0.1, 0.2 0.3, 0.9 0.9))"],
     "line_2": ["LINESTRING (0.1 0.1, 0.12 0.15, 0.2 0.3)"]
     ......
   }
   ```

   算法内部需要：

   - 针对字段名称将所设置的值进行组合，例如将`polygon_1`和`line_1`组合为一组，从而合成自己所需的格式；
   - 对于数量，通过字段的数字后缀来遍历得到，例如当外部传入`polygon_3`、`line_3`时，算法内部需要**自行通过遍历获得这第三组配置**；

   以上配置方法必须在实现时写入`config/README.md`配置文档。

4. 行人员闯入算法通常需要配置一个分析区域和一条闯入边界线，可以使用以下配置：

   ```json
   {
     "roi_type":"polygon_1;cross_line_1",
     "polygon_1": ["POLYGON ((0.1 0.1, 0.1 0.1, 0.2 0.3, 0.9 0.9))"],
     "cross_line_1": ["LINESTRING (0.1 0.1, 0.12 0.15, 0.2 0.3)"]
   }
   ```
### 如何使用jsoncpp生成和解析json字符串？
参见src/SampleAlgorithm.cpp中的使用示例
