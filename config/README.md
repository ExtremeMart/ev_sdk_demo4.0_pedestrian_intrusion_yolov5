# 示例检测算法配置文件说明

此文档是算法配置参数的使用说明

| 版本  | 修订日期 | 版本更新说明                         |
| :-----: | :--------: | :--------------------------: |
| 1.0  | 20201121 | 算法对应配置文件 |
| 2.0  | 20211121 | 增加roi_type字段 |


## 1. 算法配置参数说明

| 参数        | 说明                                                         | 是否必选 | 可否为空  |类型 |
| :-----------: | ---------------------------- | :-------------: |:-------------: | :-------------: |
|`draw_roi_area`|是否在结果图片或者视频中画出ROI感兴趣区域，`true`画，`false`不画；默认：`true`|否|否|`bool`|
|`roi_color`|ROI框的颜色，BGRA浮点型数组格式，BGR三通道的范围是`[0,255]`，第四个元素A是透明度，范围是`[0,1.0]`，值越大，画的框、文字透明度越高；默认：`[0, 255, 0, 0.4]`|否|否|`array`|
|`roi_line_thickness`|ROI线框粗细；默认：`3`|否|否|`int`|
|`roi_fill`|是否使用颜色填充ROI区域，`true`填充，`false`不填充；默认：`false`|否|否|`bool`|
|`roi_type`|算法支持的点线框类型|是|否|`string`|
|`polygon_1`|针对图片的感兴趣区域进行分析，如果没有此参数或者此参数解析错误，则ROI默认值为整张图片区域，类型：使用WKT格式表示的字符串数组|否|否|`array`|
|`draw_result`|是否画出检测目标框，`true`画，`false`不画；默认：`true`|否|否|`bool`|
|`draw_confidence`|是否将置信度画在框顶部，`true`显示，`false`不显示，默认：`true`，小数点后保留两位|否|否|`float`|
|`thresh`|检测阈值，设置越大，召回率越高，设置越小，精确率越高；范围：`[0,1]`，默认：`0.5`，推荐：`0.5`|否|否|`float`|
|`language`|显示语言，可选：zh（中文）、en（英文），默认： `en`|否|否|`string`|
|`target_rect_color`|目标框的颜色，BGRA浮点型数组格式，BGR三通道的范围是`[0,255]`，第四个元素是透明度，范围是`[0,1.0]`，值越大，画的框、文字透明度越高|否|否|`array`|
|`object_rect_line_thickness`|目标框的粗细，默认：`3`|否|否|`int`|
|`object_text_color`|目标框顶部文字的颜色，BGRA浮点型数组格式，BGR三通道的范围是`[0,255]`，第四个元素不使用，默认：`[255, 255, 255, 0]`|否|否|`array`|
|`object_text_bg_color`|目标框顶部文字的背景颜色，BGRA浮点型数组格式，BGR三通道的范围是`[0,255]`，第四个元素不使用，默认：`[50, 50, 50, 0]`|否|否|`array`|
|`object_text_size`|目标框顶部文字大小，整形，范围`[1,50]`，默认：`30`|否|否|`int`|
|`draw_warning_text`|是否画报警信息文字，`true`是，`false`否，默认：`true`|否|否|`bool`|
|`mark_text_en`|目标文字（英文），默认：`dog`|否|否|`string`|
|`mark_text_zh`|目标文字（中文），默认：`狗`|否|否|`string`|
|`warning_text_en`|报警文字（英文），默认：`WARNING!`|否|否|`string`|
|`warning_text_zh`|报警文字（中文），默认：`警告！`|否|否|`string`|
|`warning_text_size`|报警文字大小，范围`[1,50]`，默认:`30`|否|否|`array`|
|`warning_text_color`|报警文字颜色，BGRA浮点型数组格式，BGR三通道的范围是`[0,255]`，第四个元素不使用，默认：`[255, 255, 255, 0]`|否|否|`array`|
|`warning_text_bg_color`|报警文字背景颜色，BGRA浮点型数组格式，BGR三通道的范围是`[0,255]`，第四个元素不使用，默认：`[0, 0, 200, 0]`|否|否|`array`|
|`warning_text_left_top`|报警文字左上角坐标，整形数组，格式：`[x, y]`，`x`的范围`[0,width]`，`y`的范围`[0,height]`，默认：`[0, 0]`|否|否|`array`|


### 1.1 示例配置

```json
{
    "draw_roi_area": true,
    "roi_color": [255, 255, 0, 0.7],
    "roi_type":"polygon_1",
    "polygon_1": ["POLYGON((0.07878787878787878 0.1575,0.05757575757575758 0.8925,0.8484848484848485 0.9325,0.7742424242424243 0.0875,0.4257575757575758 0.0575))"],
    "roi_line_thickness": 4,
    "roi_fill": false,
    "draw_result": true,
    "draw_confidence": true,
    "thresh": 0.55,
    "language": "en",

    "target_rect_color": [0, 255, 0, 0],
    "object_rect_line_thickness": 3,
    "object_text_color": [255, 255, 255, 0],
    "object_text_bg_color": [50, 50, 50, 0],
    "object_text_size": 30,
    "mark_text_en": "dog",
    "mark_text_zh": "狗",
    "draw_warning_text": true,
    "warning_text_en": "WARNING! WARNING!",
    "warning_text_zh": "警告!",
    "warning_text_size": 30,
    "warning_text_color": [255, 255, 255, 0],
    "warning_text_bg_color": [0, 0, 200, 0],
    "warning_text_left_top": [0, 0]
}
```

## 2. ROI设置说明

#### 2.1 参数格式说明

| ROI参数类型说明                       | 是/否 |
| ------------------------------------- | ----- |
| 多个点、线、框的配置项是否使用WTK格式 | 否    |

**注**：多个点、线、框实现方式：

- 使用数组形式，如：`"polygon_1":["POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))", "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"]`。

#### 2.2 参数设置说明

| 变量名 | 用途说明                     | 类型（点/线/框）| 数量限制 | 是否可不划定（是/否） | 特别说明                           |
| ---- | ---------------- | -------|---- | -------- | -------- | ------------ |
| `roi`  | 例如：用于划定红路灯识别范围 | 框               | 无       | 是                    | 如：划定的第一个框需要在第二个框内 ||


#### 2.3 ROI设置示例

![示例ROI](sample.jpg)

## 3. 其他必要说明

其他必要的算法配置说明。

