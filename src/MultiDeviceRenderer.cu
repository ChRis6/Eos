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


#include "MultiDeviceRenderer.h"
#include "kernelWrapper.h"

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

HOST MultiDeviceRenderer::MultiDeviceRenderer(Scene* hostScene, Camera* hostCamera, int width, int height){

	int num_gpus;
	int i;
	cudaScene_t** cudaSceneArray;

	// get number of gpus on this machine
	cudaGetDeviceCount(&num_gpus);
	if( num_gpus < 1){
		fprintf(stderr, "Eos Error:No nvidia gpus found\n");
		exit(1);
	}

	printf("number of host CPUs:\t%d\n", omp_get_num_procs());
   	printf("number of CUDA devices:\t%d\n", num_gpus);

   	int ii;
   	for ( ii = 0; ii < num_gpus; ii++)
   	{
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, ii);
        printf("   %d: %s\n", ii, dprop.name);
   	}

   printf("---------------------------\n");



	// Allocate memory for gpu scene.
	// Every gpu will have a copy of the
	// the same scene,since every cudaRay
	// must have access to all geometry
	cudaSceneArray = ( cudaScene_t**) malloc( sizeof( cudaScene_t*) * num_gpus);
	if( !cudaSceneArray){
		fprintf(stderr, "Eos Error: Not enough memory on HOST\n" );
		exit(1);
	}

	for( i = 0; i < num_gpus; i++){
		cudaSceneArray[i] = NULL;
		cudaErrorCheck( cudaSetDevice(i));

		cudaSceneArray[i] = createCudaScene( hostScene);
		if( cudaSceneArray[i] == NULL){
			cudaDeviceProp dprop;
        	cudaGetDeviceProperties(&dprop, i);
        		
			fprintf(stderr, "Eos Error:Not enough memory on device %d: %s\n", i, dprop.name );
			exit(1);
		}
	}
	m_MultiGpuScene = cudaSceneArray;

	m_Width = width;
	m_Height = height;
	m_NumDevices = num_gpus;

	int pixels_per_gpu = width * height / num_gpus;

	m_DeviceImageBuffer = (char**) malloc( sizeof(char*) * num_gpus);
	

	if( !m_DeviceImageBuffer){
		fprintf(stderr, "Eos Error: Not enough memory on HOST\n" );
		exit(1);
	}

	for( i = 0; i < num_gpus; i++){
		cudaErrorCheck( cudaSetDevice(i));
		cudaErrorCheck( cudaMalloc( &m_DeviceImageBuffer[i], sizeof(char) * pixels_per_gpu * 4 )); // RGBA
	}



	m_CameraHandler = new DeviceCameraHandler[num_gpus];
	for( i = 0; i < num_gpus; i++){
		cudaErrorCheck( cudaSetDevice(i));
		m_CameraHandler[i].setDeviceCamera( hostCamera);
	}
}

HOST void MultiDeviceRenderer::renderSceneToHostBuffer( void* imageBuffer, Camera* hostCamera){

	
	//printf("Setting %d threads\n", m_NumDevices );
	omp_set_num_threads( m_NumDevices);


	int pixels_per_gpu = (m_Width * m_Height) / m_NumDevices;
	//printf("Pixels per GPU: %d\n", pixels_per_gpu );

	#pragma omp parallel
    {


    	int gpuSlice = m_Height / m_NumDevices;
    	

    	int threadsPerBlock[2];
    	int blockDim[2];

    	threadsPerBlock[0] = 16;
    	threadsPerBlock[1] = 16;

    	blockDim[0] = m_Width / threadsPerBlock[0];
    	blockDim[1] = gpuSlice / threadsPerBlock[1];

    	unsigned int cpu_thread_id = omp_get_thread_num();
    	//printf("Thread has id %d\n", cpu_thread_id );

    	cudaSetDevice( cpu_thread_id);
    	// update camera
    	m_CameraHandler[ cpu_thread_id].updateDeviceCamera( hostCamera);


    	// clear device image
    	//printf("Setting gpu %d image to zero\n", cpu_thread_id );
    	cudaMemset( m_DeviceImageBuffer[ cpu_thread_id], 0, sizeof(char) * pixels_per_gpu * 4);

    	// launch kernel
    	//printf("launching %d gpu kernel\n", cpu_thread_id );
    	rayTrace_MultiMegaKernel( m_MultiGpuScene[cpu_thread_id], m_CameraHandler[cpu_thread_id].getDeviceCamera() ,
    		                      m_Width, m_Height,
    		                      (uchar4*) m_DeviceImageBuffer[ cpu_thread_id],
	                              m_NumDevices, cpu_thread_id,
	                               blockDim, threadsPerBlock);

    	int pixel_start = pixels_per_gpu * cpu_thread_id;
    	// copy sub image back to host
    	//printf("Copying gpu %d results\n", cpu_thread_id );
    	cudaMemcpy( (char*)imageBuffer + ( pixel_start * 4), m_DeviceImageBuffer[cpu_thread_id] , sizeof(char) * 4 * pixels_per_gpu,cudaMemcpyDeviceToHost);
    }

    omp_set_num_threads( omp_get_num_procs());
}