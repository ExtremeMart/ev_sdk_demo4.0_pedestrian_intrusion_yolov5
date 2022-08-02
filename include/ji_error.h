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

#ifndef EV_JI_ERROR_H_
#define EV_JI_ERROR_H_

/**
 * @brief Enumerates ev_sdk error codes.
 *
 * @since 4.0
 */
typedef enum {
    /** No errors. */
    JISDK_RET_SUCCEED = 0,

    /** Failed.*/
    JISDK_RET_FAILED = -1,

    /** Not implemented. */
    JISDK_RET_UNUSED = -2,

    /** Invalid param. */
    JISDK_RET_INVALIDPARAMS = -3,

    /** Offline status during online verification. */
    JISDK_RET_OFFLINE = -9,

    /** Exceeded the maximum amount of requests. */
    JISDK_RET_OVERMAXQPS = -99,

    /** Unauthorized. */
    JISDK_RET_UNAUTHORIZED = -999,
    
    /** No face detection. */
    JISDK_RET_NO_FACE_DETECTED = -1000,

    /** No face lib id not exist. */
    JISDK_RET_FACE_LIB_ID_NOT_EXIST = -1001,

    /** Face id not exist. */
    JISDK_RET_FACE_ID_NOT_EXIST = -1002,

    /** Face id exist. */
    JISDK_RET_FACE_ID_EXIST = -1003,

    /** Path error. */
    JISDK_RET_PATH_ERROR = -1004,
	
	/** File error. */
	JISDK_RET_FILE_ERROR = -1005,
	
	/** Abnormal picture brightness. */
	JISDK_RET_LIGHT_ERROR = -1006,
	
	/** Imange size error. */
	JISDK_RET_IMAGE_SIZE_ERROR = -1007,
	
	/** Multiple faces. */
	JISDK_RET_MULTI_FACE_DETECTED = -1008,
	
	/** Face database id exist. */
	JISDK_RET_FACE_DB_ID_EXIST = -1009,
	
	/** The face database has exceeded the upper limit. */
	JISDK_RET_FACE_DB_OVERRUN = -1010,
	
	/** Face database format error. */
	JISDK_RET_FACE_DB_FORMAT_ERROR = -1011,
	
	/** Face database is empty. */
	JISDK_RET_FACE_DB_IS_NULL = -1012,
	
	/** Duplicate face addition. */
	JISDK_RET_FACE_REPEAD_ADD = -1013,
	
	/** Other exceptions. */
	JISDK_RET_OTHER_FAILED = -1014

}JiErrorCode;

#endif // EV_JI_ERROR_H_