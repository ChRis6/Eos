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

#ifndef _CUDA_QUALIFIERS_H
#define _CUDA_QUALIFIERS_H

#include <stdio.h>

// comment when error checking is not required
#define CHECK_FOR_CUDA_ERRORS

#ifdef __CUDACC__
	#define HOST __host__
	#define DEVICE __device__
 	#define FORCE_INLINE __forceinline__
	#define cudaErrorCheck(ans) { __cudaErrorCheck((ans), __FILE__, __LINE__); }
	inline void __cudaErrorCheck(cudaError_t code, const char *file, int line, bool abort=true)
	{
		#ifdef CHECK_FOR_CUDA_ERRORS
   		if (code != cudaSuccess) 
   		{
      		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      		if (abort) exit(code);
   		}
   		#endif
	}

#else
	#define HOST
	#define DEVICE
	#define FORCE_INLINE
	#define cudaErrorCheck(ans) 
#endif

#endif