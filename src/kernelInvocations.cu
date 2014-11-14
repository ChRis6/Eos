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

/* ================================= DEVICE FUNCTIONS ========================*/


DEVICE void traverseTreeAndStore(const Ray& ray, cudaIntersection_t* intersectionBuffer, int intersectionBufferSize, DTriangle* trianglesBuffer, int trianglesBufferSize, BvhNode* bvh, int threadID ){

    BvhNode* stackLocal[128];
    BvhNode** stack_ptr = stackLocal;
    float minDistace = 999.0f;
    float minBoxDistance = 0.0f;
    BvhNode* currNode = &bvh[0];    // bvh

    //if( currNode->aabb.intersectWithRayNew(ray) == false )
    //    return;

    minDistace = 9999.0f;
    // push null
    *stack_ptr++ = NULL;

    while(currNode != NULL){

        if( currNode->type == BVH_NODE ){

            //float leftDistance = 9999.0f;
            //float rightDistance = 9999.0f;

            /*
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
            */

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

    // prefer L1 cache
    cudaFuncSetCacheConfig("__calculateIntersections_kernel", cudaFuncCachePreferL1);
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