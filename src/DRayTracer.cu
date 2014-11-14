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

#include "DRayTracer.h"
#include "BVH.h"
#define MAX_TRACED_DEPTH 4

DEVICE glm::vec4 DRayTracer::rayTrace(DScene* scene, Camera* camera, const Ray& ray,  int depth, BvhNode** sharedStack, int threadIndex){
	DRayIntersection intersection;
	
	if( depth > MAX_TRACED_DEPTH )
		return glm::vec4(0.0f);

	if( scene->findMinDistanceIntersectionBVH(ray, intersection, sharedStack, threadIndex))
		return this->shadeIntersection(scene, ray, camera, intersection, depth);

	return glm::vec4(0.0f);
}

DEVICE glm::vec4 DRayTracer::shadeIntersection(DScene* scene, const Ray& ray, Camera* camera, DRayIntersection& intersection, int depth){
	int numLights;
	glm::vec4 finalColor(0.0f);

	numLights = scene->getNumLights();
	for( int i = 0; i < numLights; i++){
		finalColor += this->calcPhong(camera, scene->getLightSource(i), intersection);
	}
	return finalColor;
}

DEVICE glm::vec4 DRayTracer::shadeIntersectionNew(Camera* camera, DRayIntersection* intersectionBuffer, DLightSource* lights, int numLights,
												 DMaterial* materials, int numMaterials, int threadID){

	glm::vec4 finalColor(0.0f);
	for( int i = 0 ; i < numLights; i++)
		finalColor += this->calcPhongWithMaterials(camera, &lights[i], intersectionBuffer[threadID], materials, numMaterials);

	return finalColor;
}

DEVICE glm::vec4 DRayTracer::calcPhongWithMaterials(Camera* camera, DLightSource* lightSource, DRayIntersection& intersection, DMaterial* materials, int numMaterials){
	glm::vec4 color(0.0f, 0.0f, 0.0f, 0.0f);
	//glm::vec4 specularColor(0.0f, 0.0f, 0.0f, 0.0f);
	//glm::vec4 intersectionToLight;
	//glm::vec4 viewVector;
	//glm::vec4 reflectedVector;
	const DMaterial& intersectionMaterial = materials[ intersection.getIntersectionMaterialIndex()];

	const glm::vec4& intersectionPointInWorld  = intersection.getIntersectionPoint();
	const glm::vec4& intersectionNormalInWorld = intersection.getIntersectionNormal();

	
	// specular reflection
	const glm::vec4& cameraPosVec4       = glm::vec4(camera->getPosition(),1.0f);
	const glm::vec4& intersectionToLight = glm::normalize(lightSource->getPosition() - intersectionPointInWorld);
	const glm::vec4& viewVector          = glm::normalize(cameraPosVec4- intersectionPointInWorld);
	const glm::vec4& reflectedVector     = glm::normalize((2.0f * glm::dot(intersectionNormalInWorld, intersectionToLight) * intersectionNormalInWorld) - intersectionToLight);
	
	// find diffuse first
	//diffuseColor = this->findDiffuseColor(lightSource, glm::normalize(lightSource->getPosition() - intersectionPointInWorld), intersection);

	float dot = glm::dot( glm::normalize(cameraPosVec4 - intersectionPointInWorld), reflectedVector);
	if( dot > 0.0f){
		float specularTerm = glm::pow(dot, (float)intersectionMaterial.getShininess());
		color += specularTerm * lightSource->getColor() * intersectionMaterial.getSpecularColor();
	}

	return this->findDiffuseColorWithMaterials(lightSource, glm::normalize(lightSource->getPosition() - intersectionPointInWorld), intersection, materials, numMaterials) + color;

}

DEVICE glm::vec4 DRayTracer::calcPhong(Camera* camera, DLightSource* lightSource, DRayIntersection& intersection){
/*
	glm::vec4 color(0.0f, 0.0f, 0.0f, 0.0f);
	//glm::vec4 specularColor(0.0f, 0.0f, 0.0f, 0.0f);
	//glm::vec4 intersectionToLight;
	//glm::vec4 viewVector;
	//glm::vec4 reflectedVector;

	const glm::vec4& intersectionPointInWorld  = intersection.getIntersectionPoint();
	const glm::vec4& intersectionNormalInWorld = intersection.getIntersectionNormal();

	
	// specular reflection
	const glm::vec4& cameraPosVec4       = glm::vec4(camera->getPosition(),1.0f);
	const glm::vec4& intersectionToLight = glm::normalize(lightSource->getPosition() - intersectionPointInWorld);
	const glm::vec4& viewVector          = glm::normalize(cameraPosVec4- intersectionPointInWorld);
	const glm::vec4& reflectedVector     = glm::normalize((2.0f * glm::dot(intersectionNormalInWorld, intersectionToLight) * intersectionNormalInWorld) - intersectionToLight);
	
	float dot = glm::dot( glm::normalize(cameraPosVec4 - intersectionPointInWorld), reflectedVector);
	if( dot > 0.0f){
		float specularTerm = glm::pow(dot, (float)intersection.getIntersectionMaterial().getShininess());
		color += specularTerm * lightSource->getColor() * intersection.getIntersectionMaterial().getSpecularColor();
	}

	return this->findDiffuseColor(lightSource, glm::normalize(lightSource->getPosition() - intersectionPointInWorld), intersection) + color;
*/
	return glm::vec4(0.0f);
}

DEVICE glm::vec4 DRayTracer::findDiffuseColorWithMaterials(DLightSource* lightSource, const glm::vec4& intersectionToLight, DRayIntersection& intersection,
				DMaterial* materials, int numMaterials){
	
	const DMaterial& material = materials[ intersection.getIntersectionMaterialIndex()];
	float dot = glm::dot(intersectionToLight, intersection.getIntersectionNormal());
	dot = glm::max(0.0f, dot);
	return glm::vec4( dot * material.getDiffuseColor() * lightSource->getColor());
}

DEVICE glm::vec4 DRayTracer::findDiffuseColor(DLightSource* lightSource, const glm::vec4& intersectionToLight, DRayIntersection& intersection){
/*	
	const DMaterial& material = intersection.getIntersectionMaterial();
	float dot = glm::dot(intersectionToLight, intersection.getIntersectionNormal());
	dot = glm::max(0.0f, dot);
	return glm::vec4( dot * material.getDiffuseColor() * lightSource->getColor());
*/
	return glm::vec4(0.0f);
}