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

#ifndef _CUDAPRINT_H
#define _CUDAPRINT_H

#include <stdio.h>
#include "Ray.h"
#include "DScene.h"
#include "cudaStructures.h"
 
__global__ void printHelloFromGPUKernel();
__global__ void printDeviceSceneGPUKernel(DScene* d_scene);
__global__ void __debug_printCudaScene_kernel(cudaScene_t* deviceScene);
__global__ void __shuffleTest_kernel();
#endif