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




#include <algorithm>
#include "morton.h"


HOST void calcMorton(unsigned int width, unsigned int height, std::vector<unsigned int>& array){
	unsigned int i;
	unsigned int j;

	for( i = 0; i < width; i++){
		for( j = 0; j < height; j++){
			unsigned int mortonCode = EncodeMorton2(i,j);
			array.push_back(mortonCode);
		}
	}

	std::sort (array.begin(), array.end());
}

HOST void getCudaMortonBuffer(std::vector<unsigned int>& array, unsigned int** buffer, int* buffer_len){

	unsigned int* d_buffer;
	int size;

	size = array.size();

	cudaErrorCheck( cudaMalloc((void**) &d_buffer, sizeof(unsigned int) * size));
	cudaErrorCheck( cudaMemcpy(d_buffer, &array[0], sizeof(unsigned int) * size, cudaMemcpyHostToDevice));

	*buffer = d_buffer;
	*buffer_len = size;
}