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

#ifndef EV_JI_H_
#define EV_JI_H_

#include "ji_types.h"
#include "ji_error.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief SDK initialization interface.The authorization function can be implemented in the function
 * @param[in] argc Number of parameters
 * @param[in] argv Parameter array
 * @return JiErrorCode  - operation result
 */
JiErrorCode ji_init(int argc, char **argv);

/**
 * @brief SDK de initialization function
 * @return void         - operation result
 */
void ji_reinit();

/**
 * @brief Create an algorithm predictor instance.
 * 
 * @param[in] pdtype - the predictor type
 * @return void*     - the pionter of algorithm predictor instance
 */
void* ji_create_predictor(JiPredictorType pdtype);

/**
 * @brief Free an algorithm predictor instance.
 * 
 * @param[in] predictor - the predictor instance pointer
 * @return void         - operation result
 */
void ji_destroy_predictor(void *predictor);


/**
 * @brief set callback function
 * 
 * @param[in] predictor - the predictor instance pointer
 * @param[in] callback  - the the callback fucntion
 * @return JiErrorCode         - operation result
 */
JiErrorCode ji_set_callback(void *predictor, JiCallBack callback);

/**
 * @brief Picture analysis asynchronous interface.
 * 
 * @param[in]  predictor  - predictor instance
 * @param[in]  pInFrames  - input picture information array
 * @param[in]  nInCount   - picture information array size
 * @param[in]  args       - custom algorithm parameters，such as roi
 * @param[in]  userData   - user data for callbcak
 * @return JiErrorCode    - operation result
 */

JiErrorCode ji_calc_image_asyn(void* predictor, const JiImageInfo* pInFrames, const unsigned int nInCount, const char* args, void *userData);

/**
 * @brief Picture analysis synchronous interface.
 * 
 * @param[in]  predictor  - predictor instance
 * @param[in]  pInFrames  - input picture information array
 * @param[in]  nInCount   - picture information array size
 * @param[in]  args       - custom algorithm parameters，such as roi
 * @param[out] pOutFrames - output picture information array
 * @param[out] nOutCount  - output picture information array size
 * @param[out] event      - report algorithm analysis result event
 * @return JiErrorCode    - operation result
 */
JiErrorCode ji_calc_image(void* predictor, const JiImageInfo* pInFrames, const unsigned int nInCount, const char* args, 
						JiImageInfo **pOutFrames, unsigned int & nOutCount, JiEvent &event);

/**
 * @brief Update algorithm configuration.
 *
 * @param[in] predictor - predictor instance
 * @param[in] args      - custom algorithm parameters，such as roi
 * @return JiErrorCode  - operation result
 */
JiErrorCode ji_update_config(void *predictor, const char *args);

/**
 * @brief Get the sdk version.
 *
 * @param[out] pVersion - the current version
 * @return JiErrorCode  - operation result
 */
JiErrorCode ji_get_version(char *pVersion);

/**
 * @brief Create face DB.
 *
 * @param[in] predictor      - predictor instance
 * @param[in] faceDBName     - face DB name
 * @param[in] faceDBId       - face DB id 
 * @param[in] faceDBDes      - face DB describe
 * @return JiErrorCode       - operation result
 */
JiErrorCode ji_create_face_db(void *predictor, const char *faceDBName, const int faceDBId, const char *faceDBDes);

/**
 * @brief Delete face DB.
 *
 * @param[in] predictor      - predictor instance
 * @param[in] faceDBId       - face DB id 
 * @return JiErrorCode       - operation result
 */
JiErrorCode ji_delete_face_db(void *predictor, const int faceDBId);

/**
 * @brief Get face DB info.
 *
 * @param[in] predictor      - predictor instance
 * @param[in] faceDBId       - face DB id 
 * @param[out] faceDBDes     - face info
 * @return JiErrorCode       - operation result
 */
JiErrorCode ji_get_face_db_info(void *predictor, const int faceDBId, char *info);


/**
 * @brief Add face to face DB .
 *
 * @param[in] predictor       - predictor instance
 * @param[in] faceDBId        - face DB id 
 * @param[in] faceName        - face name  
 * @param[in] faceId          - face id 
 * @param[in] data            - face data 
 * @param[in] dataType        - face data type 1 jpg data, 2 image path
 * @param[out] imagePath      - output image file full path
 * @return JiErrorCode        - operation result
 */
JiErrorCode ji_face_add(void *predictor, const int faceDBId, const char *faceName, const int faceId, const char *data, const int dataType, char *imagePath);

/**
 * @brief Update face to face DB .
 *
 * @param[in] predictor       - predictor instance
 * @param[in] faceDBId        - face DB id 
 * @param[in] faceName        - face name  
 * @param[in] faceId          - face id 
 * @param[in] data            - face data 
 * @param[in] dataType        - face data type 1 jpg data, 2 image path
 * @param[out] imagePath      - output image file full path
 * @return JiErrorCode        - operation result
 */
JiErrorCode ji_face_update(void *predictor, const int faceDBId, const char *faceName, const int faceId, const char *data, const int dataType, char *imagePath);

/**
 * @brief Delete face in face DB .
 *
 * @param[in] predictor       - predictor instance
 * @param[in] faceDBId        - face DB id 
 * @param[in] faceName        - face name  
 * @param[in] faceId          - face id 
 * @return JiErrorCode        - operation result
 */
JiErrorCode ji_face_delete(void *predictor, const int faceDBId, const int faceId);


#ifdef __cplusplus
}
#endif

#endif // EV_JI_H_