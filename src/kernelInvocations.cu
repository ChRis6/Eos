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

#include "kernelInvocations.h"
#include "Ray.h"

/* ===============  KERNELS =================*/
/*
 * Simple kernel
 * one thread/pixel
 */
__global__ void __oneThreadPerPixel_kernel(){


}

__global__ void __renderToBuffer_kernel(char* buffer, unsigned int buffer_len, Camera* camera, DScene* scene, DRayTracer* rayTracer, int width, int height){
	
	
    int pi = blockIdx.x * blockDim.x + threadIdx.x;
    int pj = blockIdx.y * blockDim.y + threadIdx.y;
     

    if (pi < width && pj < height){

    	// generate ray
    	Ray ray;
  
    	//float bb = (pj - norm_height) / norm_height;
    	//float aa = (pi - norm_width) / norm_width;
        //float aa = ((2.0f * pi - width) / (float) width);
    	//float bb = ((2.0f * pj - height) / (float) height);

    	ray.setOrigin(camera->getPosition());
    	ray.setDirection( glm::normalize((((2.0f * pi - width) / (float) width) * camera->getRightVector() ) + ( ((2.0f * pj - height) / (float) height) * camera->getUpVector()) + camera->getViewingDirection()));
 		
 		// find color
 		glm::vec4 color = rayTracer->rayTrace(scene, camera, ray, 0);	// depth = 0

 		// store color
    	buffer[4 * (pi + pj * width)]      = floor(color.x == 1.0 ? 255 : fminf(color.x * 256.0f, 255.0f));
        buffer[1 +  4* (pi + pj * width)]  = floor(color.y == 1.0 ? 255 : fminf(color.y * 256.0f, 255.0f));
        buffer[2 +  4* (pi + pj * width)]  = floor(color.z == 1.0 ? 255 : fminf(color.z * 256.0f, 255.0f));
        buffer[3 +  4* (pi + pj * width)]  = (char)255;
    }     

}


/* ============ WRAPPERS ====================*/

void renderToBuffer(char* buffer, unsigned int buffer_len, Camera* camera, DScene* scene, DRayTracer* rayTracer, int blockdim[], int tpblock[], int width, int height){

	dim3 threadsPerBlock;
	dim3 numBlocks;

	threadsPerBlock.x = tpblock[0];
	threadsPerBlock.y = tpblock[1];

	numBlocks.x = blockdim[0];
	numBlocks.y = blockdim[1];

	__renderToBuffer_kernel<<<numBlocks, threadsPerBlock>>>(buffer, buffer_len, camera, scene, rayTracer, width, height);
}