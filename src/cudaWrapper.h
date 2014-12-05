/*
 * Copyright (c) 2014 Christos Papaioannou
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef _CUDAWRAPPER_H
#define _CUDAWRAPPER_H

#include "cudaStructures.h"

#ifdef __CUDACC__
	#define HOST __host__
	#define DEVICE __device__
#else
	#define HOST
	#define DEVICE
#endif

class DScene;
extern "C" void printHelloGPU();
extern "C" void printDeviceScene(DScene* d_scene);
extern "C" void resetDevice();
extern "C" void setGLDevice(int dev_id);
extern "C" void debug_printCudaScene(cudaScene_t* deviceScene);
extern "C" void shuffleTest();
extern "C" void gpuSynch();
extern "C" void cudaPreferL1Cache();

#endif