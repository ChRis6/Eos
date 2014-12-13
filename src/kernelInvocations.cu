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


#define threadID_mega (width * ( blockIdx.y * blockDim.y + threadIdx.y) + blockIdx.x * blockDim.x + threadIdx.x)


/* ===============  KERNELS =================*/

__global__ void __rayTrace_MegaKernel( cudaScene_t* deviceScene, Camera* camera, int width, int height, uchar4* imageBuffer){

    int stack[128];
    intersection_t intersection;
    glm::vec4 color(0.0f);
    float intersectionFound_dot_minDistance;
    int i;
    uchar4 ucharColor;
    
    const cudaLightSource_t* lights = deviceScene->lights;
    const cudaMaterial_t* materials = deviceScene->materials;
    const cudaBvhNode_t*  bvh = deviceScene->bvh;
    const cudaTriangle_t* triangles = deviceScene->triangles;
    const cudaTransformations_t* transformations = deviceScene->transformations;
    int* stack_ptr = stack;

    intersectionFound_dot_minDistance = 99999.0f;

    const cudaRay ray( glm::aligned_vec3(camera->getPosition()),
        glm::aligned_vec3(glm::normalize((((2.0f * (blockIdx.x * blockDim.x + threadIdx.x) - width) / (float) width) * camera->getRightVector() ) +
                        ( ((2.0f * (blockIdx.y * blockDim.y + threadIdx.y) - height) / (float) height) * camera->getUpVector())
                        + camera->getViewingDirection())));
    

    intersection.triIndex = -1;
    /*
     *
     * Tranverse BVH
     *
     */

    i = 0;
    if( __all(! (rayIntersectsCudaAABB(ray, bvh->minBoxBounds[i], bvh->maxBoxBounds[i], intersectionFound_dot_minDistance)) )){
            //write to color to global memory
        ucharColor.x = 0;
        ucharColor.y = 0;
        ucharColor.z = 0;
        ucharColor.w = 255;

        imageBuffer[threadID_mega] = ucharColor;
        return;
    }

    // push -1
    *stack_ptr++ = -1;

    while( i != -1){

        if( bvh->type[i] == BVH_NODE){

            const int leftChildIndex  = bvh->leftChildIndex[i];
            const int rightChildIndex = bvh->rightChildIndex[i];

            if( __any(rayIntersectsCudaAABB( ray, bvh->minBoxBounds[leftChildIndex], bvh->maxBoxBounds[leftChildIndex], intersectionFound_dot_minDistance))){

                if( __any(rayIntersectsCudaAABB( ray, bvh->minBoxBounds[rightChildIndex], bvh->maxBoxBounds[rightChildIndex], intersectionFound_dot_minDistance)))
                    *stack_ptr++ = rightChildIndex;
                i = leftChildIndex;
            }
            else if( __any(rayIntersectsCudaAABB( ray, bvh->minBoxBounds[rightChildIndex], bvh->maxBoxBounds[rightChildIndex], intersectionFound_dot_minDistance))){
                i = rightChildIndex;
            }
            else{
                //pop
                i = *--stack_ptr;
            }

        }
        else if( deviceScene->bvh->type[i] == BVH_LEAF ){

            intersectCudaRayWithCudaLeafRestricted(ray,// ray
                                    i ,
                                    bvh->numSurfacesEncapulated, /*bvh->surfacesIndices,*/ transformations->inverseTransformation, // bvh
                                    triangles->v1, triangles->v2, triangles->v3, // triangle vertices
                                    triangles->transformationIndex,    // triangle transformations
                                    &intersectionFound_dot_minDistance, &intersection, bvh->firstTriangleIndex[i], threadID_mega );
            // pop
            i = *--stack_ptr;
        }
    }

    // store intersection
    intersectionFound_dot_minDistance = ( (float) ( (int)intersection.triIndex != -1)); 

    //if( threadIntersection.triIndex != -1){
    const glm::vec4 intersectionPoint = intersectionFound_dot_minDistance * glm::vec4(ray.getOrigin() + intersection.baryCoords.x * ray.getDirection(), 1.0f); 
    const int materialIndex = triangles->materialIndex[ ((int) intersection.triIndex != -1) * intersection.triIndex];
    
    // find triangle normal.Interpolate vertex normals
    glm::vec4 normal = intersectionFound_dot_minDistance * glm::vec4(glm::normalize( triangles->n1[intersection.triIndex] * ( 1.0f - intersection.baryCoords.y - intersection.baryCoords.z) + (deviceScene->triangles->n2[intersection.triIndex] * intersection.baryCoords.y) + (deviceScene->triangles->n3[intersection.triIndex] * intersection.baryCoords.z)), 0.0f);
    // transform normal
    normal = transformations->inverseTransposeTransformation[ triangles->transformationIndex[ ((int) intersection.triIndex != -1) * intersection.triIndex]] * normal; 


    /*
     *
     * Shade intersection
     *
     */

    const glm::vec4& cameraPosVec4 = glm::vec4(camera->getPosition(), 1.0f);

    for( i = 0; i < lights->numLights; i++){
            
        const glm::vec4& intersectionToLight = glm::normalize( lights->positions[i] - intersectionPoint);
        const glm::vec4& reflectedVector     = glm::reflect( -intersectionToLight, normal );

        intersectionFound_dot_minDistance = glm::dot(intersectionToLight, normal);
        if( intersectionFound_dot_minDistance > 0.0f){
            color += intersectionFound_dot_minDistance * materials->diffuse[materialIndex] * lights->colors[i];
        }

        intersectionFound_dot_minDistance = glm::dot( glm::normalize(cameraPosVec4 - intersectionPoint), reflectedVector);
        if( intersectionFound_dot_minDistance > 0.0f){
            intersectionFound_dot_minDistance = glm::pow(intersectionFound_dot_minDistance, (float)materials->shininess[materialIndex]);
            color += intersectionFound_dot_minDistance * lights->colors[i] * materials->specular[materialIndex];
        }
    }


    //write to color to global memory
    ucharColor.x = floor(color.x == 1.0 ? 255 : fminf(color.x * 256.0f, 255.0f));
    ucharColor.y = floor(color.y == 1.0 ? 255 : fminf(color.y * 256.0f, 255.0f));
    ucharColor.z = floor(color.z == 1.0 ? 255 : fminf(color.z * 256.0f, 255.0f));
    ucharColor.w = 255;

    imageBuffer[threadID_mega] = ucharColor;    
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
            //intersectCudaRayWithCudaLeafRestricted(ray,
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
DEVICE FORCE_INLINE void  intersectCudaRayWithCudaLeafRestricted( const cudaRay& ray,// ray
                                    int bvhLeafIndex, int* __restrict__ numSurfacesEncapulated, int* __restrict__ surfacesIndices, glm::mat4* __restrict__ inverseTransformation, // bvh
                                    glm::aligned_vec3* __restrict__ v1, glm::aligned_vec3* __restrict__ v2, glm::aligned_vec3* __restrict__ v3, // triangle vertices
                                    int* __restrict__ triTransIndex,    // triangle transformations
                                    float* __restrict__ minDistace, intersection_t* __restrict__ threadIntersection, int threadID ){

    cudaRay localRay;
    glm::vec3 baryCoords(0.0f);

    int minTriangleIndex;
    int i;
    minTriangleIndex = -1;

    #pragma unroll 4
    for( i = 0; i < numSurfacesEncapulated[bvhLeafIndex]; i++){
        

        
        // transform Ray to local Triangle Coordinates
         
        // first get triangle index in the triangle buffer
        // every leaf has SURFACES_PER_LEAF(look at BVH.h) triangles
        // get triangle i
        const int triangleIndex = surfacesIndices[ bvhLeafIndex * SURFACES_PER_LEAF + i];

        // get Transformation index
        //int triangleTransformationIndex = deviceScene->triangles->transformationIndex[ triangleIndex];

        // transform ray origin and direction
        localRay.setOrigin( glm::aligned_vec3( inverseTransformation[ triTransIndex[ triangleIndex]] * glm::vec4(ray.getOrigin(), 1.0f)));
        localRay.setDirection( glm::aligned_vec3( inverseTransformation[ triTransIndex[ triangleIndex]] * glm::vec4(ray.getDirection(), 0.0f)));


        
        // Now intersect Triangle with local ray
        if( rayIntersectsCudaTriangle( localRay, v1[triangleIndex], v2[triangleIndex], v3[triangleIndex], baryCoords) && baryCoords.x < *minDistace){
            // intersection is found
            minTriangleIndex = triangleIndex;
            *minDistace = baryCoords.x;
            
            threadIntersection->triIndex = minTriangleIndex;
            threadIntersection->baryCoords = baryCoords;
        }// endif

    }// end for
}
*/


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



/* ============ WRAPPERS ====================*/

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