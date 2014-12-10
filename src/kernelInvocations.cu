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
#include "BVH.h"



/* ===============  KERNELS =================*/

__global__ void __rayTrace_MegaKernel( cudaScene_t* deviceScene, Camera* camera, int width, int height, uchar4* imageBuffer){

    int stack[128];
    intersection_t intersection;
    cudaLightSource_t*     lights;
    cudaMaterial_t*        materials;
    glm::vec4 color(0.0f);


    lights = deviceScene->lights;
    materials = deviceScene->materials;

    const int pi = blockIdx.x * blockDim.x + threadIdx.x;
    const int pj = blockIdx.y * blockDim.y + threadIdx.y;

    const int threadID = width * pj + pi;
    float minDistace = 99999.0f;

    
    const cudaRay ray( glm::aligned_vec3(camera->getPosition()),
        glm::aligned_vec3(glm::normalize((((2.0f * (blockIdx.x * blockDim.x + threadIdx.x) - width) / (float) width) * camera->getRightVector() ) +
                        ( ((2.0f * (blockIdx.y * blockDim.y + threadIdx.y) - height) / (float) height) * camera->getUpVector())
                        + camera->getViewingDirection())));
    
    //Ray ray;
    //ray.setOrigin(camera->getPosition());
    //ray.setDirection( glm::normalize((((2.0f * (blockIdx.x * blockDim.x + threadIdx.x) - width) / (float) width) * camera->getRightVector() ) + ( ((2.0f * (blockIdx.y * blockDim.y + threadIdx.y) - height) / (float) height) * camera->getUpVector()) + camera->getViewingDirection()));


    intersection.triIndex = -1;
    //intersection.baryCoords = glm::vec3(0.0f);

    /*
     *
     * Tranverse BVH
     *
     */

    int currNodeIndex = 0;

    int* stack_ptr = stack;

    if( __all(! (rayIntersectsCudaAABB(ray, deviceScene->bvh->minBoxBounds[ currNodeIndex], deviceScene->bvh->maxBoxBounds[currNodeIndex], minDistace)) )){
            //write to color to global memory
        uchar4 ucharColor;
        ucharColor.x = 0;
        ucharColor.y = 0;
        ucharColor.z = 0;
        ucharColor.w = 255;

        imageBuffer[threadID] = ucharColor;
        return;
    }

    // push -1
    *stack_ptr++ = -1;

    while( currNodeIndex != -1){

        if( deviceScene->bvh->type[ currNodeIndex] == BVH_NODE){

            const int leftChildIndex  = deviceScene->bvh->leftChildIndex[currNodeIndex];
            const int rightChildIndex = deviceScene->bvh->rightChildIndex[currNodeIndex];

            if( __any(rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[leftChildIndex], deviceScene->bvh->maxBoxBounds[leftChildIndex], minDistace))){

                currNodeIndex = leftChildIndex;

                if( __any(rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[rightChildIndex], deviceScene->bvh->maxBoxBounds[rightChildIndex], minDistace)))
                    *stack_ptr++ = rightChildIndex;
            }
            else if( __any(rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[rightChildIndex], deviceScene->bvh->maxBoxBounds[rightChildIndex], minDistace))){
                currNodeIndex = rightChildIndex;
            }
            else{
                //pop
                currNodeIndex = *--stack_ptr;
            }

        }
        else if( deviceScene->bvh->type[ currNodeIndex] == BVH_LEAF ){

            intersectCudaRayWithCudaLeafRestricted(ray,// ray
                                    currNodeIndex ,
                                    deviceScene->bvh->numSurfacesEncapulated, deviceScene->bvh->surfacesIndices, deviceScene->transformations->inverseTransformation, // bvh
                                    deviceScene->triangles->v1,deviceScene->triangles->v2, deviceScene->triangles->v3, // triangle vertices
                                    deviceScene->triangles->transformationIndex,    // triangle transformations
                                    &minDistace, &intersection, threadID );
            // pop
            currNodeIndex = *--stack_ptr;
        }
    }

    // store intersection
    float intersectionFound = ( (float) ( (int)intersection.triIndex != -1)); 

    //if( threadIntersection.triIndex != -1){
    const glm::vec4 intersectionPoint = intersectionFound * glm::vec4(ray.getOrigin() + intersection.baryCoords.x * ray.getDirection(), 1.0f); 
    const int materialIndex = deviceScene->triangles->materialIndex[ ((int) intersection.triIndex != -1) * intersection.triIndex];
    
    // find triangle normal.Interpolate vertex normals
    glm::vec4 normal = intersectionFound * glm::vec4(glm::normalize(deviceScene->triangles->n1[intersection.triIndex] * ( 1.0f - intersection.baryCoords.y - intersection.baryCoords.z) + (deviceScene->triangles->n2[intersection.triIndex] * intersection.baryCoords.y) + (deviceScene->triangles->n3[intersection.triIndex] * intersection.baryCoords.z)), 0.0f);
    // transform normal
    normal = deviceScene->transformations->inverseTransposeTransformation[ deviceScene->triangles->transformationIndex[ ((int) intersection.triIndex != -1) * intersection.triIndex]] * normal; 


    /*
     *
     * Shade intersection
     *
     */


    const glm::vec4& cameraPosVec4 = glm::vec4(camera->getPosition(), 1.0f);
    const glm::vec4& intersectionPointInWorld  = intersectionPoint;
    const glm::vec4& intersectionNormalInWorld = normal;

    float dot;
    const int numLights = lights->numLights;
    for( int i = 0; i < numLights; i++){
            
        const glm::vec4& intersectionToLight = glm::normalize(lights->positions[i] - intersectionPointInWorld);
        const glm::vec4& reflectedVector     = glm::reflect( -intersectionToLight, intersectionNormalInWorld );

        dot = glm::dot(intersectionToLight, intersectionNormalInWorld);
        if( dot > 0.0f){
            color += dot * materials->diffuse[materialIndex] * lights->colors[i];
        }

        dot = glm::dot( glm::normalize(cameraPosVec4 - intersectionPointInWorld), reflectedVector);
        if( dot > 0.0f){
            float specularTerm = glm::pow(dot, (float)materials->shininess[materialIndex]);
            color += specularTerm * lights->colors[i] * materials->specular[materialIndex];
        }
    }


    //write to color to global memory
    uchar4 ucharColor;
    ucharColor.x = floor(color.x == 1.0 ? 255 : fminf(color.x * 256.0f, 255.0f));
    ucharColor.y = floor(color.y == 1.0 ? 255 : fminf(color.y * 256.0f, 255.0f));
    ucharColor.z = floor(color.z == 1.0 ? 255 : fminf(color.z * 256.0f, 255.0f));
    ucharColor.w = 255;

    imageBuffer[threadID] = ucharColor;    
}

/**** NOT WORKING *****/
__global__ void __rayTrace_WarpShuffle_MegaKernel( cudaScene_t* deviceScene, Camera* camera, int width, int height, uchar4* imageBuffer){


    int stack[128];
    intersection_t intersection;
    cudaLightSource_t*     lights;
    cudaMaterial_t*        materials;
    glm::vec4 color(0.0f);


    lights = deviceScene->lights;
    materials = deviceScene->materials;

    const int y = blockIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadID = y * width + x; 


    float minDistace = 99999.0f;

    
    const Ray ray( camera->getPosition(),
        glm::normalize((((2.0f * x - width) / (float) width) * camera->getRightVector() ) +
                        ( ((2.0f * y - height) / (float) height) * camera->getUpVector())
                        + camera->getViewingDirection()));

    intersection.triIndex = -1;
    intersection.baryCoords = glm::vec3(0.0f);

    /*
     *
     * Tranverse BVH
     *
     */

    int currNodeIndex = 0;

    int* stack_ptr = stack;

    int leftVotes;
    int rightVotes;

    if( __all(! (rayIntersectsCudaAABB(ray, deviceScene->bvh->minBoxBounds[ currNodeIndex], deviceScene->bvh->maxBoxBounds[currNodeIndex], minDistace)) )){
            //write to color to global memory
        uchar4 ucharColor;
        ucharColor.x = 0;
        ucharColor.y = 0;
        ucharColor.z = 0;
        ucharColor.w = 255;

        imageBuffer[threadID] = ucharColor;
        return;
    }

    // push -1
    *stack_ptr++ = -1;

    while( currNodeIndex != -1){

        if( deviceScene->bvh->type[ currNodeIndex] == BVH_NODE){

            const int leftChildIndex  = deviceScene->bvh->leftChildIndex[currNodeIndex];
            const int rightChildIndex = deviceScene->bvh->rightChildIndex[currNodeIndex];

            leftVotes  = rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[leftChildIndex], deviceScene->bvh->maxBoxBounds[leftChildIndex], minDistace);
            rightVotes = rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[rightChildIndex], deviceScene->bvh->maxBoxBounds[rightChildIndex], minDistace);


            // warp votes for left and  right bvh children
            for( int i = 1; i < warpSize; i *=2)
                leftVotes += __shfl_xor( leftVotes, i);

            for( int i = 1; i < warpSize; i *=2)
                rightVotes += __shfl_xor( rightVotes, i);

            /*
             * Choose next bvh node
             *
             * Note that all threads have the same values
             * in variables leftVotes and rightVotes,which means that
             * they will all choose the same next node.As a result
             * there is no divergence.
             */
            if( leftVotes > rightVotes){
                currNodeIndex = leftChildIndex;
                if( rightVotes > 0)
                    *stack_ptr++ = rightChildIndex;
            }
            else if( rightVotes > 0){
                currNodeIndex = rightChildIndex;
            }
            else {
                currNodeIndex = *--stack_ptr;

            }

            leftVotes = 0;
            rightVotes = 0;
        }
        else if( deviceScene->bvh->type[ currNodeIndex] == BVH_LEAF ){

            intersectRayWithCudaLeafRestricted(ray,// ray
                                    currNodeIndex ,
                                    deviceScene->bvh->numSurfacesEncapulated, deviceScene->bvh->surfacesIndices, deviceScene->transformations->inverseTransformation, // bvh
                                    deviceScene->triangles->v1,deviceScene->triangles->v2, deviceScene->triangles->v3, // triangle vertices
                                    deviceScene->triangles->transformationIndex,    // triangle transformations
                                    &minDistace, &intersection, threadID );
            // pop
            currNodeIndex = *--stack_ptr;
        }
    }

    // store intersection
    float intersectionFound = ( (float) ( (int)intersection.triIndex != -1)); 

    //if( threadIntersection.triIndex != -1){
    const glm::vec4 intersectionPoint = intersectionFound * glm::vec4(ray.getOrigin() + intersection.baryCoords.x * ray.getDirection(), 1.0f); 
    const int materialIndex = deviceScene->triangles->materialIndex[ ((int) intersection.triIndex != -1) * intersection.triIndex];
    
    // find triangle normal.Interpolate vertex normals
    glm::vec4 normal = intersectionFound * glm::vec4(glm::normalize(deviceScene->triangles->n1[intersection.triIndex] * ( 1.0f - intersection.baryCoords.y - intersection.baryCoords.z) + (deviceScene->triangles->n2[intersection.triIndex] * intersection.baryCoords.y) + (deviceScene->triangles->n3[intersection.triIndex] * intersection.baryCoords.z)), 0.0f);
    // transform normal
    normal = deviceScene->transformations->inverseTransposeTransformation[ deviceScene->triangles->transformationIndex[ ((int) intersection.triIndex != -1) * intersection.triIndex]] * normal; 


    /*
     *
     * Shade intersection
     *
     */


    const glm::vec4& cameraPosVec4 = glm::vec4(camera->getPosition(), 1.0f);
    const glm::vec4& intersectionPointInWorld  = intersectionPoint;
    const glm::vec4& intersectionNormalInWorld = normal;

    float dot;
    const int numLights = lights->numLights;
    for( int i = 0; i < numLights; i++){
            
        const glm::vec4& intersectionToLight = glm::normalize(lights->positions[i] - intersectionPointInWorld);
        const glm::vec4& reflectedVector     = glm::reflect( -intersectionToLight, intersectionNormalInWorld );

        dot = glm::dot(intersectionToLight, intersectionNormalInWorld);
        if( dot > 0.0f){
            color += dot * materials->diffuse[materialIndex] * lights->colors[i];
        }

        dot = glm::dot( glm::normalize(cameraPosVec4 - intersectionPointInWorld), reflectedVector);
        if( dot > 0.0f){
            float specularTerm = glm::pow(dot, (float)materials->shininess[materialIndex]);
            color += specularTerm * lights->colors[i] * materials->specular[materialIndex];
        }
    }


    //write to color to global memory
    uchar4 ucharColor;
    ucharColor.x = floor(color.x == 1.0 ? 255 : fminf(color.x * 256.0f, 255.0f));
    ucharColor.y = floor(color.y == 1.0 ? 255 : fminf(color.y * 256.0f, 255.0f));
    ucharColor.z = floor(color.z == 1.0 ? 255 : fminf(color.z * 256.0f, 255.0f));
    ucharColor.w = 255;

    imageBuffer[threadID] = ucharColor; 


}






/*
 * Simple kernel
 * one thread/pixel
 */
__global__ void __oneThreadPerPixel_kernel(){


}

__global__ void __renderToBuffer_kernel(char* buffer, unsigned int buffer_len, Camera* camera, DScene* scene, DRayTracer* rayTracer, int width, int height){
	
	/*
    int pi = (blockIdx.x * blockDim.x + threadIdx.x);
    int pj = (blockIdx.y * blockDim.y + threadIdx.y);
    extern __shared__ BvhNode* bvh_stack[];
    BvhNode** bvh_stack_ptr = bvh_stack;

    int threadStackIndex = (threadIdx.y * blockDim.x + threadIdx.x ) * 30;

    if ( pi < width && pj < height){

    	// generate ray
    	Ray ray;
  
    	//float bb = (pj - norm_height) / norm_height;
    	//float aa = (pi - norm_width) / norm_width;
        //float aa = ((2.0f * pi - width) / (float) width);
    	//float bb = ((2.0f * pj - height) / (float) height);

    	ray.setOrigin(camera->getPosition());
    	ray.setDirection( glm::normalize((((2.0f * pi - width) / (float) width) * camera->getRightVector() ) + ( ((2.0f * pj - height) / (float) height) * camera->getUpVector()) + camera->getViewingDirection()));
 		
 		// find color
 		glm::vec4 color = rayTracer->rayTrace(scene, camera, ray, 0, bvh_stack_ptr, threadStackIndex);	// depth = 0

 		// store color
    	buffer[4 * (pi + pj * width)]      = floor(color.x == 1.0 ? 255 : fminf(color.x * 256.0f, 255.0f));
        buffer[1 +  4* (pi + pj * width)]  = floor(color.y == 1.0 ? 255 : fminf(color.y * 256.0f, 255.0f));
        buffer[2 +  4* (pi + pj * width)]  = floor(color.z == 1.0 ? 255 : fminf(color.z * 256.0f, 255.0f));
        buffer[3 +  4* (pi + pj * width)]  = (char)255;
    } 
    */    

}

__global__ void __calculateIntersections_kernel(Camera* camera,
                                                cudaIntersection_t* intersectionBuffer, int intersectionBufferSize, 
                                                DTriangle* trianglesBuffer, int trianglesBufferSize,
                                                BvhNode* bvh, int width, int height){

    int pi = (blockIdx.x * blockDim.x + threadIdx.x);
    int pj = (blockIdx.y * blockDim.y + threadIdx.y);

    int threadID = width * pj + pi;

    if ( pi < width && pj < height){
        Ray ray;

        ray.setOrigin(camera->getPosition());
        ray.setDirection( glm::normalize((((2.0f * pi - width) / (float) width) * camera->getRightVector() ) + ( ((2.0f * pj - height) / (float) height) * camera->getUpVector()) + camera->getViewingDirection()));

        traverseTreeAndStore( ray, intersectionBuffer, intersectionBufferSize, trianglesBuffer, trianglesBufferSize, bvh, threadID);
    }

}

__global__ void __shadeIntersectionsToBuffer_kernel(uchar4* imageBuffer, unsigned int imageSize, DRayTracer* rayTracer, Camera* camera,
                                                    DLightSource* lights, int numLights,
                                                    cudaIntersection_t* intersectionBuffer, int intersectionBufferSize,
                                                    DMaterial* materialsBuffer, int materialsBufferSize, 
                                                    int width, int height){


    //int pi = (blockIdx.x * blockDim.x + threadIdx.x);
    //int pj = (blockIdx.y * blockDim.y + threadIdx.y);
    
    int threadID = width * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x);
    if( (blockIdx.x * blockDim.x + threadIdx.x) < width && (blockIdx.y * blockDim.y + threadIdx.y) < height ){
    
        //glm::vec4 color = rayTracer->shadeIntersectionNew(camera, intersectionBuffer, lights, numLights, materialsBuffer, materialsBufferSize, threadID);
        const glm::vec4& cameraPosVec4       = glm::vec4(camera->getPosition(),1.0f);
        const DMaterial& intersectionMaterial = materialsBuffer[ intersectionBuffer->materialsIndices[threadID]];
        const glm::vec4& intersectionPointInWorld  = intersectionBuffer->points[threadID];
        const glm::vec4& intersectionNormalInWorld = intersectionBuffer->normals[threadID];
        
        glm::vec4 color(0.0f);
        float dot;
        
        //#pragma unroll 2
        for( int i = 0 ; i < numLights; i++){
            
            const glm::vec4& intersectionToLight = glm::normalize(lights[i].getPosition() - intersectionPointInWorld);
            const glm::vec4& viewVector          = glm::normalize(cameraPosVec4- intersectionPointInWorld);
            //const glm::vec4& reflectedVector     = glm::normalize((2.0f * glm::dot(intersectionNormalInWorld, intersectionToLight) * intersectionNormalInWorld) - intersectionToLight);
            const glm::vec4& reflectedVector     = glm::reflect( -intersectionToLight, intersectionNormalInWorld );
            // calculate diffuse color
            dot = glm::dot(intersectionToLight, intersectionNormalInWorld);
            if( dot > 0.0f){
                color += dot * intersectionMaterial.getDiffuseColor() * lights[i].getColor();

                dot = glm::dot( glm::normalize(cameraPosVec4 - intersectionPointInWorld), reflectedVector);
                if( dot > 0.0f){
                    float specularTerm = glm::pow(dot, (float)intersectionMaterial.getShininess());
                    color += specularTerm * lights[i].getColor() * intersectionMaterial.getSpecularColor();
                }
            }
        }


        uchar4 ucharColor;
        ucharColor.x = floor(color.x == 1.0 ? 255 : fminf(color.x * 256.0f, 255.0f));
        ucharColor.y = floor(color.y == 1.0 ? 255 : fminf(color.y * 256.0f, 255.0f));
        ucharColor.z = floor(color.z == 1.0 ? 255 : fminf(color.z * 256.0f, 255.0f));
        ucharColor.w = 255;

        imageBuffer[threadID] = ucharColor;

    }
}

/**
 * NEW cudaScene kernels.
 *
 * cudaScene_t* is memory friendly for gpus
 *
 */





__global__ void __calculateCudaSceneIntersections_kernel( cudaScene_t* deviceScene, Camera* camera, cudaIntersection_t* intersectionBuffer, int width, int height){


    __shared__ int sharedStack[128];
    __shared__ int sharedBvhNodeIndex;
    __shared__ int sharedVotes[2];


    volatile int pi = blockIdx.x * blockDim.x + threadIdx.x;
    volatile int pj = blockIdx.y * blockDim.y + threadIdx.y;



    int threadID = width * pj + pi;

    int threadBlockID = threadIdx.x + threadIdx.y;
    //threadBlockID = threadBlockID / 32;

    
    if( threadBlockID == 0){
        sharedStack[0] = -1;      // push -1 
        sharedBvhNodeIndex = 0;   // start at root
        sharedVotes[0] = 0;
        sharedVotes[1] = 0;
        //printf("Thread %d in block initialized shared memory\n", threadID);
    }

    //__syncthreads();


    if ( (blockIdx.x * blockDim.x + threadIdx.x) < width && (blockIdx.y * blockDim.y + threadIdx.y) < height){
        Ray ray;

        ray.setOrigin(camera->getPosition());
        ray.setDirection( glm::normalize((((2.0f * (blockIdx.x * blockDim.x + threadIdx.x) - width) / (float) width) * camera->getRightVector() ) + ( ((2.0f * (blockIdx.y * blockDim.y + threadIdx.y) - height) / (float) height) * camera->getUpVector()) + camera->getViewingDirection()));
              

        //traverseCudaTreeAndStore( deviceScene, ray, intersectionBuffer, threadID);
        traverseCudaTreeAndStoreSharedStack( sharedStack, &sharedBvhNodeIndex, sharedVotes, deviceScene, ray, intersectionBuffer, threadID, threadBlockID);
        //traverseCudaTreeAndStoreNew( deviceScene, ray, intersectionBuffer, threadID);
    }

}


__global__ void __shadeCudaSceneIntersections_kernel( cudaScene_t* deviceScene, Camera* camera, cudaIntersection_t* intersectionBuffer, int width, int height, uchar4* imageBuffer){

    cudaLightSource_t*     lights;
    cudaMaterial_t*        materials;
    glm::vec4 color(0.0f);


    lights = deviceScene->lights;
    materials = deviceScene->materials;




    int threadID = width * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x);
    int numLights = lights->numLights;

    if( (blockIdx.x * blockDim.x + threadIdx.x) < width && (blockIdx.y * blockDim.y + threadIdx.y) < height ){

        const glm::vec4& cameraPosVec4       = glm::vec4(camera->getPosition(),1.0f);
        const glm::vec4& intersectionPointInWorld  = intersectionBuffer->points[threadID];
        const glm::vec4& intersectionNormalInWorld = intersectionBuffer->normals[threadID];
        int materialIndex = intersectionBuffer->materialsIndices[threadID];

        float dot;
        for( int i = 0; i < numLights; i++){
            
            const glm::vec4& intersectionToLight = glm::normalize(lights->positions[i] - intersectionPointInWorld);
            //const glm::vec4& viewVector          = glm::normalize(cameraPosVec4- intersectionPointInWorld);
            const glm::vec4& reflectedVector     = glm::reflect( -intersectionToLight, intersectionNormalInWorld );

            dot = glm::dot(intersectionToLight, intersectionNormalInWorld);
            if( dot > 0.0f){
                color += dot * materials->diffuse[materialIndex] * lights->colors[i];

            }

            dot = glm::dot( glm::normalize(cameraPosVec4 - intersectionPointInWorld), reflectedVector);
            if( dot > 0.0f){
                float specularTerm = glm::pow(dot, (float)materials->shininess[materialIndex]);
                color += specularTerm * lights->colors[i] * materials->specular[materialIndex];
            }

        }

        uchar4 ucharColor;
        ucharColor.x = floor(color.x == 1.0 ? 255 : fminf(color.x * 256.0f, 255.0f));
        ucharColor.y = floor(color.y == 1.0 ? 255 : fminf(color.y * 256.0f, 255.0f));
        ucharColor.z = floor(color.z == 1.0 ? 255 : fminf(color.z * 256.0f, 255.0f));
        ucharColor.w = 255;

        imageBuffer[threadID] = ucharColor;
    }

}

DEVICE void traverseCudaTreeAndStoreSharedStack( int* sharedStack, int* sharedCurrNodeIndex, int* sharedVotes,
                                                cudaScene_t* deviceScene, const Ray& ray, cudaIntersection_t* intersectionBuffer, int threadID, int threadBlockID){

    /*
     * Intersect ray with bvh root.If all threads miss root's aabb then return.
     * even if  one thread intersects root, then traverse
     */
    //int queue[128];
    int stackLocal[128];
    int currNodeIndex = 0;
    //int front;
    //int rear;
    //int leftChildVote;
    //int rightChildVote;
    glm::vec4 localNormal(0.0f);

    int* stack_ptr = stackLocal;
    float minDistace = 9999.0f;
    intersection_t threadIntersection;

    threadIntersection.triIndex = -1;

    
    //int intersectsRoot = rayIntersectsCudaAABB(ray, deviceScene->bvh->minBoxBounds[ currNodeIndex], deviceScene->bvh->maxBoxBounds[currNodeIndex]);


    if( __all(! (rayIntersectsCudaAABB(ray, deviceScene->bvh->minBoxBounds[ currNodeIndex], deviceScene->bvh->maxBoxBounds[currNodeIndex], minDistace)) ))
        return;
    
    //front = rear = 0;
    //queue[rear++] = deviceScene->bvh->leftChildIndex[0];
    //queue[rear++] = deviceScene->bvh->rightChildIndex[0];

    *stack_ptr++ = -1;

    while( currNodeIndex != -1){
    //while( front != rear){
        //leftChildVote = 0;
        //rightChildVote = 0;

        //currNodeIndex = queue[front++];

        if( deviceScene->bvh->type[ currNodeIndex] == BVH_NODE){

            int leftChildIndex;
            int rightChildIndex;

            leftChildIndex  = deviceScene->bvh->leftChildIndex[currNodeIndex];
            rightChildIndex = deviceScene->bvh->rightChildIndex[currNodeIndex];

            //int leftChildIntersected  = rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[leftChildIndex], deviceScene->bvh->maxBoxBounds[leftChildIndex], 0.0f);
            //int rightChildIntersected = rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[rightChildIndex], deviceScene->bvh->maxBoxBounds[rightChildIndex], 0.0f);

            /*
            // warp sum reduce
            // sm >= 3.0
            #pragma unroll
            for (int mask = warpSize/2; mask > 0; mask /= 2) 
                leftChildVote += __shfl_xor(leftChildVote, mask);

            #pragma unroll
            for (int mask = warpSize/2; mask > 0; mask /= 2) 
                rightChildVote += __shfl_xor( rightChildVote, mask);

            // broadcast to all
            //leftChildVote  = __shfl(leftChildVote, 0);
            //rightChildVote = __shfl(rightChildVote, 0);

        
            if( leftChildVote >= rightChildVote && leftChildVote > 0){
                currNodeIndex = leftChildIndex;
                if( rightChildVote)
                    *stack_ptr++ = rightChildIndex;

            }
            else if( rightChildVote > leftChildVote && rightChildVote > 0 ){
                currNodeIndex = rightChildIndex;
                if( leftChildVote)
                    *stack_ptr++ = leftChildIndex;
            }
            else{
                // pop
                currNodeIndex = *--stack_ptr;

            }

            */

            if( __any(rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[leftChildIndex], deviceScene->bvh->maxBoxBounds[leftChildIndex], minDistace))){

                currNodeIndex = leftChildIndex;
                //queue[rear++] = deviceScene->bvh->leftChildIndex[currNodeIndex];

                if( __any(rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[rightChildIndex], deviceScene->bvh->maxBoxBounds[rightChildIndex], minDistace)))
                    *stack_ptr++ = rightChildIndex;
                    //queue[rear++] = deviceScene->bvh->rightChildIndex[currNodeIndex];
            }
            else if( __any(rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[rightChildIndex], deviceScene->bvh->maxBoxBounds[rightChildIndex], minDistace))){
                currNodeIndex = rightChildIndex;
                //queue[rear++] = deviceScene->bvh->rightChildIndex[currNodeIndex];
            }
            else{
                //pop
                currNodeIndex = *--stack_ptr;
            }

 

        }
        else if( deviceScene->bvh->type[ currNodeIndex] == BVH_LEAF ){
            //printf("Warp reached leaf\n");
            // this is a leaf
            //intersectRayWithCudaLeaf( ray, deviceScene, currNodeIndex, &minDistace, &threadIntersection, threadID);
            intersectRayWithCudaLeafRestricted(ray,// ray
                                    currNodeIndex ,
                                    deviceScene->bvh->numSurfacesEncapulated, deviceScene->bvh->surfacesIndices, deviceScene->transformations->inverseTransformation, // bvh
                                    deviceScene->triangles->v1,deviceScene->triangles->v2, deviceScene->triangles->v3, // triangle vertices
                                    deviceScene->triangles->transformationIndex,    // triangle transformations
                                    &minDistace, &threadIntersection, threadID );
            // pop
            currNodeIndex = *--stack_ptr;

        }
    }

    float intersectionFound = ( (float) ( (int)threadIntersection.triIndex != -1)); 

    //if( threadIntersection.triIndex != -1){
    intersectionBuffer->points[threadID] = intersectionFound * glm::vec4(ray.getOrigin() + threadIntersection.baryCoords.x * ray.getDirection(), 1.0f); 
    intersectionBuffer->materialsIndices[threadID] = deviceScene->triangles->materialIndex[ ((int) threadIntersection.triIndex != -1) * threadIntersection.triIndex];
        
    localNormal = intersectionFound * glm::vec4(glm::normalize(deviceScene->triangles->n1[threadIntersection.triIndex] * ( 1.0f - threadIntersection.baryCoords.y - threadIntersection.baryCoords.z) + (deviceScene->triangles->n2[threadIntersection.triIndex] * threadIntersection.baryCoords.y) + (deviceScene->triangles->n3[threadIntersection.triIndex] * threadIntersection.baryCoords.z)), 0.0f);
    intersectionBuffer->normals[threadID] = deviceScene->transformations->inverseTransposeTransformation[ deviceScene->triangles->transformationIndex[ ((int) threadIntersection.triIndex != -1) * threadIntersection.triIndex]] * localNormal; 
    //}    

}

DEVICE void traverseCudaTreeAndStoreNew( cudaScene_t* deviceScene, const Ray& ray, cudaIntersection_t* intersectionBuffer, int threadID){

    int stack[128];
    int stackIndex;
    int currNodeIndex = 0;
    float minDistace = 99999.0f;
    glm::vec4 localNormal(0.0f);
    intersection_t threadIntersection;

    threadIntersection.triIndex = -1;

    stackIndex = 1;

    if( __all(! (rayIntersectsCudaAABB(ray, deviceScene->bvh->minBoxBounds[0], deviceScene->bvh->maxBoxBounds[0], minDistace)) ))
        return;

    while( stackIndex > 0){

        while(1){

            if( deviceScene->bvh->type[ currNodeIndex] == BVH_LEAF){

                intersectRayWithCudaLeafRestricted(ray,// ray
                                    currNodeIndex ,
                                    deviceScene->bvh->numSurfacesEncapulated, deviceScene->bvh->surfacesIndices, deviceScene->transformations->inverseTransformation, // bvh
                                    deviceScene->triangles->v1,deviceScene->triangles->v2, deviceScene->triangles->v3, // triangle vertices
                                    deviceScene->triangles->transformationIndex,    // triangle transformations
                                    &minDistace, &threadIntersection, threadID );
                break;
            }
            else if( deviceScene->bvh->type[ currNodeIndex] == BVH_NODE){

                int leftChildIndex  = deviceScene->bvh->leftChildIndex[currNodeIndex];
                int rightChildIndex = deviceScene->bvh->rightChildIndex[currNodeIndex];

                int child0 = rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[leftChildIndex], deviceScene->bvh->maxBoxBounds[leftChildIndex], minDistace);
                int child1 = rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[rightChildIndex], deviceScene->bvh->maxBoxBounds[rightChildIndex], minDistace);

                if( child0 && child1){
                    currNodeIndex = leftChildIndex;
                    stack[stackIndex++] = rightChildIndex;
                }
                else if( child0){
                    currNodeIndex = leftChildIndex;
                }
                else if( child1){
                    currNodeIndex = rightChildIndex;
                }
                else 
                    break;

            }
        }// end for

        stackIndex--;
        currNodeIndex = stack[stackIndex];


    }



    float intersectionFound = ( (float) ( (int)threadIntersection.triIndex != -1)); 

    intersectionBuffer->points[threadID] = intersectionFound * glm::vec4(ray.getOrigin() + threadIntersection.baryCoords.x * ray.getDirection(), 1.0f); 
    intersectionBuffer->materialsIndices[threadID] = deviceScene->triangles->materialIndex[ ((int) threadIntersection.triIndex != -1) * threadIntersection.triIndex];
        
    localNormal = intersectionFound * glm::vec4(glm::normalize(deviceScene->triangles->n1[threadIntersection.triIndex] * ( 1.0f - threadIntersection.baryCoords.y - threadIntersection.baryCoords.z) + (deviceScene->triangles->n2[threadIntersection.triIndex] * threadIntersection.baryCoords.y) + (deviceScene->triangles->n3[threadIntersection.triIndex] * threadIntersection.baryCoords.z)), 0.0f);
    intersectionBuffer->normals[threadID] = deviceScene->transformations->inverseTransposeTransformation[ deviceScene->triangles->transformationIndex[ ((int) threadIntersection.triIndex != -1) * threadIntersection.triIndex]] * localNormal;  
}






/* ================================= DEVICE FUNCTIONS ========================*/
DEVICE void traverseCudaTreeAndStore( cudaScene_t* deviceScene, const Ray& ray, cudaIntersection_t* intersectionBuffer, int threadID){
    int stackLocal[128];
    int* stack_ptr = stackLocal;
    float minDistace = 99999.0f;
    int currentBvhNodeIndex;
    float minBoxDistance = 99999.0f;
    //cudaBvhNode_t* bvh = deviceScene->bvh;

    currentBvhNodeIndex = 0; // start at root
    glm::vec4 localNormal(0.0f);
    intersection_t threadIntersection;

    threadIntersection.triIndex = -1;

    // check root
    if( ( rayIntersectsCudaAABB(ray, deviceScene->bvh->minBoxBounds[currentBvhNodeIndex], deviceScene->bvh->maxBoxBounds[currentBvhNodeIndex], minBoxDistance) ) == false )
        return;

    // push -1
    *stack_ptr++ = -1;

    while( currentBvhNodeIndex != -1){

        if( deviceScene->bvh->type[currentBvhNodeIndex] == BVH_NODE){

            int leftChildIndex;
            int rightChildIndex;

            leftChildIndex  = deviceScene->bvh->leftChildIndex[currentBvhNodeIndex];
            rightChildIndex = deviceScene->bvh->rightChildIndex[currentBvhNodeIndex];

            // intersect both aabbs
            int leftChildIntersected = rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[deviceScene->bvh->leftChildIndex[currentBvhNodeIndex]], deviceScene->bvh->maxBoxBounds[deviceScene->bvh->leftChildIndex[currentBvhNodeIndex]], minBoxDistance);
            int rightChildIntersected = rayIntersectsCudaAABB( ray, deviceScene->bvh->minBoxBounds[deviceScene->bvh->rightChildIndex[currentBvhNodeIndex]], deviceScene->bvh->maxBoxBounds[deviceScene->bvh->rightChildIndex[currentBvhNodeIndex]], minBoxDistance); 

            if( leftChildIntersected){
                // always chose left child first
                currentBvhNodeIndex = deviceScene->bvh->leftChildIndex[currentBvhNodeIndex];
                if( rightChildIntersected)
                    // push right child
                    *stack_ptr++ = rightChildIndex;
            }
            else if( rightChildIntersected){
                currentBvhNodeIndex = rightChildIndex;
            }
            else{
                // no intersection.POP next node from stack
                currentBvhNodeIndex = *--stack_ptr;

            }
            //__syncthreads();

        }
        else if(deviceScene->bvh->type[currentBvhNodeIndex] == BVH_LEAF) {
            // reached a leaf
           
            //intersectRayWithCudaLeaf( ray, deviceScene, currentBvhNodeIndex, &minDistace, &threadIntersection, threadID);
            intersectRayWithCudaLeafRestricted(ray,// ray
                                    currentBvhNodeIndex ,
                                    deviceScene->bvh->numSurfacesEncapulated, deviceScene->bvh->surfacesIndices, deviceScene->transformations->inverseTransformation, // bvh
                                    deviceScene->triangles->v1,deviceScene->triangles->v2, deviceScene->triangles->v3, // triangle vertices
                                    deviceScene->triangles->transformationIndex,    // triangle transformations
                                    &minDistace, &threadIntersection, threadID );

            minBoxDistance = minDistace;
            // pop 
            currentBvhNodeIndex = *--stack_ptr;
           
        }
    }

    if( threadIntersection.triIndex != -1){
        printf("Thread %d stores an intersection at global memory\n", threadID);
        intersectionBuffer->points[threadID] = glm::vec4(ray.getOrigin() + threadIntersection.baryCoords.x * ray.getDirection(), 1.0f); 
        intersectionBuffer->materialsIndices[threadID] = deviceScene->triangles->materialIndex[threadIntersection.triIndex];
        
        localNormal =  glm::vec4(glm::normalize(deviceScene->triangles->n1[threadIntersection.triIndex] * ( 1.0f - threadIntersection.baryCoords.y - threadIntersection.baryCoords.z) + (deviceScene->triangles->n2[threadIntersection.triIndex] * threadIntersection.baryCoords.y) + (deviceScene->triangles->n3[threadIntersection.triIndex] * threadIntersection.baryCoords.z)), 0.0f);
        intersectionBuffer->normals[threadID] = deviceScene->transformations->inverseTransposeTransformation[ deviceScene->triangles->transformationIndex[ threadIntersection.triIndex]] * localNormal; 
    }

    //intersectionBuffer->points[threadID]  = deviceScene->transformations->transformation[ deviceScene->triangles->transformationIndex[ minTriangleIndex]] * localIntersectionPoint;
    //intersectionBuffer->normals[threadID] = deviceScene->transformations->inverseTransposeTransformation[ deviceScene->triangles->transformationIndex[ minTriangleIndex]] * localIntersectionNormal;
    //intersectionBuffer->materialsIndices[threadID] = deviceScene->triangles->materialIndex[minTriangleIndex];
}
DEVICE FORCE_INLINE void  intersectCudaRayWithCudaLeafRestricted( const cudaRay& ray,// ray
                                    int bvhLeafIndex, int* __restrict__ numSurfacesEncapulated, int* __restrict__ surfacesIndices, glm::mat4* __restrict__ inverseTransformation, // bvh
                                    glm::aligned_vec3* __restrict__ v1, glm::aligned_vec3* __restrict__ v2, glm::aligned_vec3* __restrict__ v3, // triangle vertices
                                    int* __restrict__ triTransIndex,    // triangle transformations
                                    float* __restrict__ minDistace, intersection_t* __restrict__ threadIntersection, int threadID ){

    cudaRay localRay;
    glm::vec3 baryCoords(0.0f);
    //cudaBvhNode_t* bvh;
    //cudaTransformations_t* transformations;
    //cudaTriangle_t*        triangles;
    
    //glm::vec4 localIntersectionPoint;
    //glm::vec4 localIntersectionNormal;
    int minTriangleIndex;
    int i;
    //bvh = deviceScene->bvh;
    //transformations = deviceScene->transformations;
    //triangles = deviceScene->triangles;


    //numTrianglesInLeaf = bvh->numSurfacesEncapulated[bvhLeafIndex];
    minTriangleIndex = -1;

    #pragma unroll 2
    for( i = 0; i < numSurfacesEncapulated[bvhLeafIndex]; i++){
        

        /*
         * transform Ray to local Triangle Coordinates
         */

        // first get triangle index in the triangle buffer
        // every leaf has SURFACES_PER_LEAF(look at BVH.h) triangles
        // get triangle i
        int triangleIndex = surfacesIndices[ bvhLeafIndex * SURFACES_PER_LEAF + i];

        // get Transformation index
        //int triangleTransformationIndex = deviceScene->triangles->transformationIndex[ triangleIndex];

        // transform ray origin and direction
        localRay.setOrigin( glm::aligned_vec3( inverseTransformation[ triTransIndex[ triangleIndex]] * glm::vec4(ray.getOrigin(), 1.0f)));
        localRay.setDirection( glm::aligned_vec3( inverseTransformation[ triTransIndex[ triangleIndex]] * glm::vec4(ray.getDirection(), 0.0f)));


        /*
         * Now intersect Triangle with local ray
         */
        //bool triangleIntersected = rayIntersectsCudaTriangle( localRay, triangles->v1[triangleIndex], triangles->v2[triangleIndex], triangles->v3[triangleIndex], baryCoords);
        
        if( rayIntersectsCudaTriangle( localRay, v1[triangleIndex], v2[triangleIndex], v3[triangleIndex], baryCoords) && baryCoords.x < *minDistace){
            // intersection is found
            minTriangleIndex = triangleIndex;
            *minDistace = baryCoords.x;

            //localIntersectionPoint  = glm::vec4(localRay.getOrigin() + baryCoords.x * localRay.getDirection(), 1.0f);
            //localIntersectionNormal = glm::vec4(glm::normalize(deviceScene->triangles->n1[triangleIndex] * ( 1.0f - baryCoords.y - baryCoords.z) + (deviceScene->triangles->n2[triangleIndex] * baryCoords.y) + (deviceScene->triangles->n3[triangleIndex] * baryCoords.z)), 0.0f);             
            threadIntersection->triIndex = minTriangleIndex;
            threadIntersection->baryCoords = baryCoords;
        }// endif

    }// end for




}



DEVICE FORCE_INLINE void intersectRayWithCudaLeafRestricted( const Ray& ray,// ray
                                    int bvhLeafIndex, int* __restrict__ numSurfacesEncapulated, int* __restrict__ surfacesIndices, glm::mat4* __restrict__ inverseTransformation, // bvh
                                    glm::vec3* __restrict__ v1, glm::vec3* __restrict__ v2, glm::vec3* __restrict__ v3, // triangle vertices
                                    int* __restrict__ triTransIndex,    // triangle transformations
                                    float* __restrict__ minDistace, intersection_t* __restrict__ threadIntersection, int threadID ){

    //int numTrianglesInLeaf;
    Ray localRay;
    glm::vec3 baryCoords(0.0f);
    //cudaBvhNode_t* bvh;
    //cudaTransformations_t* transformations;
    //cudaTriangle_t*        triangles;
    
    //glm::vec4 localIntersectionPoint;
    //glm::vec4 localIntersectionNormal;
    int minTriangleIndex;
    int i;
    //bvh = deviceScene->bvh;
    //transformations = deviceScene->transformations;
    //triangles = deviceScene->triangles;


    //numTrianglesInLeaf = bvh->numSurfacesEncapulated[bvhLeafIndex];
    minTriangleIndex = -1;

    #pragma unroll 2
    for( i = 0; i < numSurfacesEncapulated[bvhLeafIndex]; i++){
        

        /*
         * transform Ray to local Triangle Coordinates
         */

        // first get triangle index in the triangle buffer
        // every leaf has SURFACES_PER_LEAF(look at BVH.h) triangles
        // get triangle i
        int triangleIndex = surfacesIndices[ bvhLeafIndex * SURFACES_PER_LEAF + i];

        // get Transformation index
        //int triangleTransformationIndex = deviceScene->triangles->transformationIndex[ triangleIndex];

        // transform ray origin and direction
        localRay.setOrigin( glm::vec3( inverseTransformation[ triTransIndex[ triangleIndex]] * glm::vec4(ray.getOrigin(), 1.0f)));
        localRay.setDirection( glm::vec3( inverseTransformation[ triTransIndex[ triangleIndex]] * glm::vec4(ray.getDirection(), 0.0f)));


        /*
         * Now intersect Triangle with local ray
         */
        //bool triangleIntersected = rayIntersectsCudaTriangle( localRay, triangles->v1[triangleIndex], triangles->v2[triangleIndex], triangles->v3[triangleIndex], baryCoords);
        
        if( rayIntersectsCudaTriangle( localRay, v1[triangleIndex], v2[triangleIndex], v3[triangleIndex], baryCoords) && baryCoords.x < *minDistace){
            // intersection is found
            minTriangleIndex = triangleIndex;
            *minDistace = baryCoords.x;

            //localIntersectionPoint  = glm::vec4(localRay.getOrigin() + baryCoords.x * localRay.getDirection(), 1.0f);
            //localIntersectionNormal = glm::vec4(glm::normalize(deviceScene->triangles->n1[triangleIndex] * ( 1.0f - baryCoords.y - baryCoords.z) + (deviceScene->triangles->n2[triangleIndex] * baryCoords.y) + (deviceScene->triangles->n3[triangleIndex] * baryCoords.z)), 0.0f);             
            threadIntersection->triIndex = minTriangleIndex;
            threadIntersection->baryCoords = baryCoords;
        }// endif

    }// end for




}
DEVICE void intersectRayWithCudaLeaf( const Ray& ray, cudaScene_t* __restrict__  deviceScene, int bvhLeafIndex, float* __restrict__  minDistace, intersection_t* __restrict__  threadIntersection , int threadID){

    //int numTrianglesInLeaf;
    Ray localRay;
    glm::vec3 baryCoords(0.0f);
    //cudaBvhNode_t* bvh;
    //cudaTransformations_t* transformations;
    //cudaTriangle_t*        triangles;
    
    //glm::vec4 localIntersectionPoint;
    //glm::vec4 localIntersectionNormal;
    int minTriangleIndex;
    int i;
    //bvh = deviceScene->bvh;
    //transformations = deviceScene->transformations;
    //triangles = deviceScene->triangles;


    //numTrianglesInLeaf = bvh->numSurfacesEncapulated[bvhLeafIndex];
    minTriangleIndex = -1;

    #pragma unroll 2
    for( i = 0; i < deviceScene->bvh->numSurfacesEncapulated[bvhLeafIndex]; i++){
        

        /*
         * transform Ray to local Triangle Coordinates
         */

        // first get triangle index in the triangle buffer
        // every leaf has SURFACES_PER_LEAF(look at BVH.h) triangles
        // get triangle i
        int triangleIndex = deviceScene->bvh->surfacesIndices[ bvhLeafIndex * SURFACES_PER_LEAF + i];

        // get Transformation index
        //int triangleTransformationIndex = deviceScene->triangles->transformationIndex[ triangleIndex];

        // transform ray origin and direction
        localRay.setOrigin( glm::vec3( deviceScene->transformations->inverseTransformation[ deviceScene->triangles->transformationIndex[ triangleIndex]] * glm::vec4(ray.getOrigin(), 1.0f)));
        localRay.setDirection( glm::vec3( deviceScene->transformations->inverseTransformation[ deviceScene->triangles->transformationIndex[ triangleIndex]] * glm::vec4(ray.getDirection(), 0.0f)));


        /*
         * Now intersect Triangle with local ray
         */
        //bool triangleIntersected = rayIntersectsCudaTriangle( localRay, triangles->v1[triangleIndex], triangles->v2[triangleIndex], triangles->v3[triangleIndex], baryCoords);
        
        if( rayIntersectsCudaTriangle( localRay, deviceScene->triangles->v1[triangleIndex], deviceScene->triangles->v2[triangleIndex], deviceScene->triangles->v3[triangleIndex], baryCoords) && baryCoords.x < *minDistace){
            // intersection is found
            minTriangleIndex = triangleIndex;
            *minDistace = baryCoords.x;

            //localIntersectionPoint  = glm::vec4(localRay.getOrigin() + baryCoords.x * localRay.getDirection(), 1.0f);
            //localIntersectionNormal = glm::vec4(glm::normalize(deviceScene->triangles->n1[triangleIndex] * ( 1.0f - baryCoords.y - baryCoords.z) + (deviceScene->triangles->n2[triangleIndex] * baryCoords.y) + (deviceScene->triangles->n3[triangleIndex] * baryCoords.z)), 0.0f);             
            threadIntersection->triIndex = minTriangleIndex;
            threadIntersection->baryCoords = baryCoords;
        }// endif

    }// end for
}

/*
DEVICE bool rayIntersectsCudaTriangle( const Ray& ray, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, glm::vec3& baryCoords){


    const glm::vec3& P = glm::cross(ray.getDirection(), v3 - v1);
    float det = glm::dot(v2 - v1, P);
    if (det > -0.00001f && det < 0.00001f)
        return false;

    det = 1.0f / det;
    const glm::vec3& T = ray.getOrigin() - v1;
    const glm::vec3& Q = glm::cross(T, v2 - v1);
    
    baryCoords.x = glm::dot(v3 - v1, Q) * det;
    baryCoords.y = glm::dot(T, P) * det;
    baryCoords.z = glm::dot(ray.getDirection(), Q) * det;

    if ((baryCoords.x < 0.0f) || (baryCoords.y < 0.0f || baryCoords.y > 1.0f) || ( baryCoords.z < 0.0f || baryCoords.y + baryCoords.z > 1.0f) )
        return false;
    
    return true;

}
*/

/*
DEVICE bool rayIntersectsCudaAABB(const Ray& ray, const glm::vec4& minBoxBounds, const glm::vec4& maxBoxBounds){
    
   glm::vec4 tmin = (minBoxBounds - glm::vec4(ray.getOrigin(), 1.0f)) * glm::vec4( ray.getInvDirection(), 0.0f);
   glm::vec4 tmax = (maxBoxBounds - glm::vec4(ray.getOrigin(), 1.0f)) * glm::vec4( ray.getInvDirection(), 0.0f);
   
   glm::vec4 real_min = glm::min(tmin, tmax);
   glm::vec4 real_max = glm::max(tmin, tmax);
   
   float minmax = fminf( fminf(real_max.x, real_max.y), real_max.z);
   float maxmin = fmaxf( fmaxf(real_min.x, real_min.y), real_min.z);
    
   return ( minmax >= maxmin);
}
*/



DEVICE void traverseTreeAndStore(const Ray& ray, cudaIntersection_t* intersectionBuffer, int intersectionBufferSize, DTriangle* trianglesBuffer, int trianglesBufferSize, BvhNode* bvh, int threadID ){

    BvhNode* stackLocal[128];
    BvhNode** stack_ptr = stackLocal;
    float minDistace = 999.0f;
    float minBoxDistance = 0.0f;
    BvhNode* currNode = &bvh[0];    // bvh



    if( currNode->aabb.intersectWithRayNew(ray) == false )
        return;

    minDistace = 9999.0f;
    // push null
    *stack_ptr++ = NULL;

    while(currNode != NULL){

        if( currNode->type == BVH_NODE ){

            //float leftDistance = 9999.0f;
            //float rightDistance = 9999.0f;

            
            BvhNode* leftChild  = &bvh[currNode->leftChildIndex];   
            BvhNode* rightChild = &bvh[currNode->rightChildIndex];

            bool leftChildIntersected = leftChild->aabb.intersectWithRayOptimized(ray, 1e-3, 9999.0f);
            bool rightChildIntersected = rightChild->aabb.intersectWithRayOptimized(ray, 1e-3, 9999.0f);
            //bool rightChildIntersected = rightChild->aabb.intersectWithRayNew(ray, rightDistance);
            //bool leftChildIntersected = leftChild->aabb.intersectWithRayNew(ray, leftDistance);

            if( leftChildIntersected){
                currNode = leftChild;
                if( rightChildIntersected)
                    // push right child to stack
                    *stack_ptr++ = rightChild;
            }
            else if(rightChildIntersected){
                currNode = rightChild;
            }
            else{ // none of  the children hit the ray. POP stack
                currNode = *--stack_ptr;
            }
            

            /*
            if( currNode->aabb.intersectWithRayOptimized(ray, 0.001, 999.0f) ){

                
                if( ray.m_sign[currNode->splitAxis] ){
                    //push left child
                    *stack_ptr++ = &bvh[currNode->leftChildIndex];
                    currNode = &bvh[currNode->rightChildIndex];
                }
                else{
                    *stack_ptr++ = &bvh[currNode->rightChildIndex];
                    currNode = &bvh[currNode->leftChildIndex];
                }

            }
            else{
                // pop
                currNode = *--stack_ptr;
            }
            */
        }
        else{
            intersectRayWithLeafNode(ray, currNode, intersectionBuffer, minDistace, trianglesBuffer, threadID);
            minBoxDistance = minDistace;
            // pop 
            currNode = *--stack_ptr;
        }

    }
}

DEVICE bool intersectRayWithLeafNode(const Ray& ray, BvhNode* node, cudaIntersection_t* intersection, float& distance, DTriangle* triangles, int threadID){
    Ray localRay;
    DTriangle* minTri = NULL;

    for( int i = 0; i < node->numSurfacesEncapulated; i++){
        int triIndex = node->surfacesIndices[i];

        DTriangle* tri = &triangles[triIndex];

        // transform ray to local coordinates
        localRay.setOrigin(glm::vec3( tri->getInverseTrasformation() * glm::vec4(ray.getOrigin(), 1.0f)));
        localRay.setDirection(glm::vec3( tri->getInverseTrasformation() * glm::vec4(ray.getDirection(), 0.0f)));

        if( tri->hit(localRay, intersection, distance, threadID) )
            minTri = tri;
        
    }

    if( minTri != NULL ){
        //intersection->setIntersectionPoint(minTri->getTransformation() * intersection->getIntersectionPoint());
        //intersection->setIntersectionNormal(minTri->getInverseTransposeTransformation() * intersection->getIntersectionNormal());
        intersection->points[threadID]  = minTri->getTransformation() * intersection->points[threadID];
        intersection->normals[threadID] = minTri->getInverseTransposeTransformation() * intersection->normals[threadID];
        return true;
    }
    return false;


}

/* ============ WRAPPERS ====================*/

void renderToBuffer(char* buffer, unsigned int buffer_len, Camera* camera, DScene* scene, DRayTracer* rayTracer, int blockdim[], int tpblock[], int width, int height){

	dim3 threadsPerBlock;
	dim3 numBlocks;

	threadsPerBlock.x = tpblock[0];
	threadsPerBlock.y = tpblock[1];

	numBlocks.x = blockdim[0];
	numBlocks.y = blockdim[1];
    // prefer L1 cache
    cudaFuncSetCacheConfig("__renderToBuffer_kernel", cudaFuncCachePreferL1);
	__renderToBuffer_kernel<<<numBlocks, threadsPerBlock, 30 >>>(buffer, buffer_len, camera, scene, rayTracer, width, height);
}

void calculateIntersections(Camera* camera, cudaIntersection_t* intersectionBuffer, int intersectionBufferSize,
                            DTriangle* trianglesBuffer, int trianglesBufferSize, BvhNode* bvh,
                            int width, int height, int blockdim[], int tpblock[]){

    dim3 threadsPerBlock;
    dim3 numBlocks;

    threadsPerBlock.x = tpblock[0];
    threadsPerBlock.y = tpblock[1];

    numBlocks.x = blockdim[0];
    numBlocks.y = blockdim[1];

    __calculateIntersections_kernel<<< numBlocks, threadsPerBlock>>>( camera,
                                                                    intersectionBuffer, intersectionBufferSize,
                                                                    trianglesBuffer, trianglesBufferSize,bvh, width, height);

}

void shadeIntersectionsToBuffer(uchar4* imageBuffer, unsigned int imageSize, DRayTracer* rayTracer,
                                Camera* camera, DLightSource* lights, int numLights,
                                cudaIntersection_t* intersectionBuffer, int intersectionBufferSize,
                                DMaterial* materialsBuffer, int materialsBufferSize, 
                                int width, int height, int blockdim[], int tpblockp[]){




    dim3 threadsPerBlock;
    dim3 numBlocks;

    threadsPerBlock.x = tpblockp[0];
    threadsPerBlock.y = tpblockp[1];

    numBlocks.x = blockdim[0];
    numBlocks.y = blockdim[1];

 __shadeIntersectionsToBuffer_kernel<<< numBlocks, threadsPerBlock>>>(imageBuffer, imageSize, rayTracer, camera,
                                    lights, numLights,
                                    intersectionBuffer,intersectionBufferSize,
                                    materialsBuffer, materialsBufferSize, 
                                    width, height);

}


void calculateCudaSceneIntersections( cudaScene_t* deviceScene, Camera* camera, cudaIntersection_t* intersectionBuffer, int width, int height,
                                                 int blockdim[], int tpblock[]){

    dim3 threadsPerBlock;
    dim3 numBlocks;

    threadsPerBlock.x = tpblock[0];
    threadsPerBlock.y = tpblock[1];

    numBlocks.x = blockdim[0];
    numBlocks.y = blockdim[1];

    __calculateCudaSceneIntersections_kernel<<< numBlocks, threadsPerBlock>>>( deviceScene, camera, intersectionBuffer, width, height);

}

void shadeCudaSceneIntersections( cudaScene_t* deviceScene, Camera* camera, cudaIntersection_t* intersectionBuffer, int width, int height, uchar4* imageBuffer,
                                             int blockdim[], int tpblock[]){

    dim3 threadsPerBlock;
    dim3 numBlocks;

    threadsPerBlock.x = tpblock[0];
    threadsPerBlock.y = tpblock[1];

    numBlocks.x = blockdim[0];
    numBlocks.y = blockdim[1];

    __shadeCudaSceneIntersections_kernel<<<numBlocks, threadsPerBlock>>>( deviceScene, camera, intersectionBuffer, width, height, imageBuffer);

}

void rayTrace_MegaKernel( cudaScene_t* deviceScene, Camera* camera, int width, int height, uchar4* imageBuffer, int blockdim[], int tpblock[]){

    dim3 threadsPerBlock;
    dim3 numBlocks;

    threadsPerBlock.x = tpblock[0];
    threadsPerBlock.y = tpblock[1];

    numBlocks.x = blockdim[0];
    numBlocks.y = blockdim[1];

    __rayTrace_MegaKernel<<<numBlocks, threadsPerBlock>>>( deviceScene, camera, width, height, imageBuffer);
}


void rayTrace_WarpShuffle_MegaKernel( cudaScene_t* deviceScene, Camera* camera, int width, int height, uchar4* imageBuffer, int blockdim[], int tpblock){

    int threadsPerBlock;
    dim3 numBlocks;

    threadsPerBlock = tpblock;

    numBlocks.x = blockdim[0];
    numBlocks.y = blockdim[1];


    __rayTrace_WarpShuffle_MegaKernel<<<numBlocks, threadsPerBlock>>>( deviceScene, camera, width, height, imageBuffer);
}