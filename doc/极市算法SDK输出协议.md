# 极市算法SDK输出协议


[toc]

## 1. 概要

此文档是极视角算法SDK输出的制定协议，所有算法SDK的输出应按照这一协议定义输出。

| 更新时间  | 版本 |    备注     |
| :-------: | :--: | :---------: |
| 2021-02-1 | 1.0  | 发布1.0版本 |

协议主要基于几个考量因素制定标准：

- 满足业务需求；

- 能够进行有效的算法精度测试；

- 输出数据清晰易懂；

- 新算法能够依据此协议制定合理的输出格式。

## 2. 格式标准

### 2.1 基础规范

数据格式必须遵守[JSON标准](http://json.cn/wiki.html)。

#### 2.1.1 命名规范

- 命名采用蛇形命名法：所有命名使用小写字母和下划线表示；

#### 2.1.2 数值规范

- 所有键含义为**是/否**的数值必须使用`true`、`false`，不能使用0、1等数字表示；
- 所有与置信度相关的数值类型必须是`float`，且保留小数点后三位；

#### 2.1.3 结构规范

- 最外层为对象类型；
- 最外层对象**必须存在**的键：`model_data`、`algorithm_data`，且对应值类型必须为对象类型，其中：
  - `algorithm_data`表示业务输出，是对原始模型输出进行过滤、修改后满足业务需求的数据；
  - `model_data`表示模型输出的原始数据，用于评估原始算法模型的精度。

### 2.2 常用数据结构规范

#### 2.2.1 使用矩形框表示的目标位置

- 坐标使用以图像横向为x轴、纵向为y轴、左上顶点为原点的直角坐标系表示法；

- 使用左上顶点坐标、矩形框宽、矩形框高表示单个目标框，格式规范：

  ```json
  {
      "x": "Number（int）：坐标值",
      "y": "Number（int）：坐标值",
      "width": "Number（int）：矩形框宽",
      "height": "Number（int）：矩形框高"
  }
  ```

- 矩形框左上顶点坐标、宽、高分别使用`(x, y)`、`width`、`height`表示。

### 2.3 业务输出数据规范

#### 2.3.1 基础规范

业务输出相关的数据必须包含在`algorithm_data`键下。

- 对于报警类算法，必须存在键`is_alert`、`target_info`；
- 对于非报警类算法，必须存在键`target_info`。

#### 2.3.2 报警状态

- `is_alert`：布尔类型，可选，表示当前数据是否是报警状态，其中`true`表示报警，`false`表示不报警；

#### 2.3.3 详细业务信息

业务信息必须使用键`target_info`存储。

- `target_info`，存储业务数据，要求：
  - 当信息为空时，值为`[]`；
  - 当信息不为空时，为算法输出的具体业务信息，可以为对象类型和数组类型，对于不同的算法类别，具有不同的规范，参考以下针对每种类别算法的详细定义。
- `target_count`：整型，可选，当`target_info`值为数组类型时，表示数组的长度；
  - 如果存在多个目标数量，必须以`target_count`作为前缀命名，如`target_count_clothes`，`target_count_no_clothes`；

- 其他业务所需字段按照2.1、2.2、2.3规范进行添加。

#### 2.3.4 输出json格式示例

**报警**

```json
{
    "algorithm_data": {
        "is_alert": true,
        "target_count_hat": 1,
        "target_count_head": 1,
        "target_info": [
            {
                "x": 543,
                "y": 154,
                "width": 37,
                "height": 54,
                "name": "hat_white",
                "confidence": 0.867
            },
            {
                "x": 543,
                "y": 154,
                "width": 37,
                "height": 54,
                "name": "head",
                "confidence": 0.867
            }
        ]
    },
    "model_data": {
        "objects": [
            {
                "x": 543,
                "y": 154,
                "width": 37,
                "height": 54,
                "name": "hat_white",
                "confidence": 0.867
            },
            {
                "x": 543,
                "y": 154,
                "width": 37,
                "height": 54,
                "name": "head",
                "confidence": 0.867
            }
        ]
    }
}
```

**非报警**

```json
{
    "algorithm_data": {
        "is_alert": false,
        "target_count_head": 0,
        "target_count_hat": 0,
        "target_info": []
    },
    "model_data": {
        "objects":[
            {
                "x": 543,
                "y": 154,
                "width": 37,
                "height": 54,
                "name": "hat_white",
                "confidence": 0.867
            },
            {
                "x": 543,
                "y": 154,
                "width": 37,
                "height": 54,
                "name": "head",
                "confidence": 0.867
            }
        ]
    }
}
```
