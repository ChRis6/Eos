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

#include "DeviceSceneHandler.h"


/**
 * Copy entire scene from host to device memory
 *
 * BVH is not copied now.
 * returns: pointer to device allocated DScene class
 */

HOST DScene* DeviceSceneHandler::createDeviceScene(Scene* h_scene){
	
	DScene* d_scene = NULL;
	DScene* h_DScene;
	DLightSource* d_LightsArray;
	DTriangle* d_TriangleArray;
	int numLights;
	int numTriangles;

	// create a temp copy of DScene on host
	h_DScene = new DScene;
	//h_DScene->m_UsingBVH = false;

	cudaErrorCheck( cudaMalloc((void**)&d_scene, sizeof(DScene)) );
	if(!d_scene)
		return NULL;

	numLights = h_scene->getNumLightSources();
	numTriangles = h_scene->getNumSurfaces();

	// set number of triangles and lights 
	h_DScene->m_NumLights = numLights;
	h_DScene->m_NumTriangles = numTriangles;

	// create buffer of DLightSource objects on host
	// and then copy to Device memory
	DLightSource* h_DLightSources = new DLightSource[numLights];
	
	cudaErrorCheck( cudaMalloc((void**)&d_LightsArray, sizeof(DLightSource) * numLights));
	if(!d_LightsArray){
		cudaErrorCheck( cudaFree(d_scene));
		delete h_DLightSources;
		return NULL;
	}
	
	for( int i = 0; i < numLights; i++){
		const LightSource* h_Light = h_scene->getLightSource(i);
		
		h_DLightSources[i].m_Position = h_Light->getPosition();
		h_DLightSources[i].m_Color = h_Light->getLightColor();
	}
	
	// maybe transfer later ???
	cudaErrorCheck( cudaMemcpy(d_LightsArray, h_DLightSources, sizeof(DLightSource) * numLights, cudaMemcpyHostToDevice));
	// ATTENTION: d_LightsArray points to GPU memory.DONT DEREFERENCE ON HOST
	h_DScene->m_Lights = d_LightsArray;

	// Now copy triangles to device memory

	DTriangle* h_DTriangles = new DTriangle[numTriangles];
	for(int i = 0 ; i < numTriangles; i++){
		
		Surface* h_Surface = h_scene->getSurface(i);

		Triangle* h_Triangle = dynamic_cast<Triangle*>(h_Surface);
		
		if( !h_Triangle){
			// this surface is not a Triangle.
			// Only scenes containing Triangles will be copied to gpu.
			// cleanup and return null
			cudaErrorCheck( cudaFree(d_scene));
			cudaErrorCheck( cudaFree(d_LightsArray));
			delete h_DLightSources;
			delete h_DTriangles;
			return NULL;
		}
		//vertices
		h_DTriangles[i].m_V1 = h_Triangle->m_V1;
		h_DTriangles[i].m_V2 = h_Triangle->m_V2;
		h_DTriangles[i].m_V3 = h_Triangle->m_V3;
		// normals
		h_DTriangles[i].m_N1 = h_Triangle->m_N1;
		h_DTriangles[i].m_N2 = h_Triangle->m_N2;
		h_DTriangles[i].m_N3 = h_Triangle->m_N3;
		
		// copy transformations
		h_DTriangles[i].m_Transformation   = h_Triangle->transformation();
		h_DTriangles[i].m_Inverse          = h_Triangle->getInverseTransformation();
		h_DTriangles[i].m_InverseTranspose = h_Triangle->getInverseTransposeTransformation();
		// material
		DMaterial h_DMaterial;
		const Material& h_TriangleMaterial = h_Triangle->getMaterial(); 
		
		h_DMaterial.m_Diffuse           = h_TriangleMaterial.getDiffuseColor();
		h_DMaterial.m_Specular          = h_TriangleMaterial.getSpecularColor();
		h_DMaterial.m_AmbientIntensity  = h_TriangleMaterial.getAmbientIntensity();
		h_DMaterial.m_Reflectivity      = h_TriangleMaterial.getReflectiveIntensity();
		h_DMaterial.m_shininess         = h_TriangleMaterial.getShininess();

		h_DTriangles[i].m_Material = h_DMaterial;  
	}

	cudaErrorCheck( cudaMalloc((void**)&d_TriangleArray, sizeof(DTriangle) * numTriangles));
	cudaErrorCheck( cudaMemcpy(d_TriangleArray, h_DTriangles, sizeof(DTriangle) * numTriangles, cudaMemcpyHostToDevice));
	h_DScene->m_Triangles = d_TriangleArray;


	// is  there a bvh ?
	if( h_scene->isUsingBvh()){

		//h_DScene->m_UsingBVH = true;
		const BVH& sceneBVH = h_scene->getBVH();
		int numNodes   = sceneBVH.getNodesBufferSize();
		BvhNode* h_bvh = sceneBVH.getNodesBuffer();

		BvhNode* h_Dbvh = new BvhNode[numNodes];
		for( int i = 0; i < numNodes; i++){
			h_Dbvh[i] = h_bvh[i];
		}

		// allocate memory on device
		BvhNode* d_bvh;
		cudaErrorCheck( cudaMalloc((void**)&d_bvh, sizeof(BvhNode) * numNodes));
		cudaErrorCheck( cudaMemcpy(d_bvh, h_Dbvh, sizeof(BvhNode) * numNodes, cudaMemcpyHostToDevice));

		h_DScene->m_BvhBuffer = d_bvh;
		//h_DScene->m_BvhBufferSize = numNodes;

		delete h_Dbvh;
	}

	// also copy h_DScene
	cudaErrorCheck( cudaMemcpy(d_scene, h_DScene, sizeof(DScene), cudaMemcpyHostToDevice)); 

	//delete h_DScene;
	m_HostScene = h_DScene;
	delete h_DLightSources;
	delete h_DTriangles;
	return d_scene;
}

HOST void DeviceSceneHandler::freeDeviceScene(){
	
	DScene* d_scene = this->getDeviceSceneDevicePointer();

	cudaErrorCheck( cudaFree(d_scene->m_Triangles));
	cudaErrorCheck( cudaFree(d_scene->m_Lights));
	cudaErrorCheck( cudaFree(d_scene));
	delete m_HostScene;	
}

DScene* DeviceSceneHandler::getDeviceSceneHostPointer(){
	return m_HostScene;
}
HOST DScene* DeviceSceneHandler::getDeviceSceneDevicePointer(){
	return m_DeviceScene;
}