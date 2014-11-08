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


DEVICE glm::vec4 DRayTracer::rayTrace(DScene* scene, Camera* camera, const Ray& ray,  int depth){
	DRayIntersection intersection;
	
	if( depth > this->getTracedDepth() )
		return glm::vec4(0.0f);
	// find itersection
	if( scene->isUsingBVH() ){
		if( scene->findMinDistanceIntersectionBVH(ray, intersection))
			return this->shadeIntersection(scene, ray, camera, intersection, depth);

	}
	else{
		if( scene->findMinDistanceIntersectionLinear( ray, intersection)){

			return this->shadeIntersection(scene, ray, camera, intersection, depth);
		}
	}
	return glm::vec4(0.0f);
}

DEVICE glm::vec4 DRayTracer::shadeIntersection(DScene* scene, const Ray& ray, Camera* camera, DRayIntersection& intersection, int depth){
	int numLights;
	int i;
	glm::vec4 finalColor(0.0f);

	numLights = scene->getNumLights();
	for( i = 0; i < numLights; i++){
		finalColor += this->calcPhong(camera, scene->getLightSource(i), intersection);
	}
	return finalColor;
}

DEVICE glm::vec4 DRayTracer::calcPhong(Camera* camera, DLightSource* lightSource, DRayIntersection& intersection){
	glm::vec4 color(0.0f, 0.0f, 0.0f, 0.0f);
	//glm::vec4 specularColor(0.0f, 0.0f, 0.0f, 0.0f);
	//glm::vec4 intersectionToLight;
	//glm::vec4 viewVector;
	//glm::vec4 reflectedVector;

	//glm::vec4 intersectionPointInWorld  = glm::vec4(intersection.getIntersectionPoint() , 1.0f);
	//glm::vec4 intersectionNormalInWorld = glm::vec4(intersection.getIntersectionNormal(), 0.0f);

	
	// specular reflection
	glm::vec4 intersectionToLight = glm::normalize(lightSource->getPosition() - glm::vec4(intersection.getIntersectionPoint() , 1.0f));
	//viewVector          = glm::normalize(glm::vec4(camera->getPosition(),1.0f) - glm::vec4(intersection.getIntersectionPoint() , 1.0f));
	//reflectedVector     = glm::normalize((2.0f * glm::dot(glm::vec4(intersection.getIntersectionNormal(), 0.0f), intersectionToLight) * glm::vec4(intersection.getIntersectionNormal(), 0.0f)) - intersectionToLight);
	
	// find diffuse first
	//diffuseColor = this->findDiffuseColor(lightSource, glm::normalize(lightSource->getPosition() - glm::vec4(intersection.getIntersectionPoint() , 1.0f)), intersection);

	float dot = glm::dot( glm::normalize(glm::vec4(camera->getPosition(),1.0f) - glm::vec4(intersection.getIntersectionPoint() , 1.0f)), glm::normalize((2.0f * glm::dot(glm::vec4(intersection.getIntersectionNormal(), 0.0f), intersectionToLight) * glm::vec4(intersection.getIntersectionNormal(), 0.0f)) - intersectionToLight));
	if( dot > 0.0f){
		float specularTerm = glm::pow(dot, (float)intersection.getIntersectionMaterial().getShininess());
		color += specularTerm * lightSource->getColor() * intersection.getIntersectionMaterial().getSpecularColor();
	}

	return this->findDiffuseColor(lightSource, glm::normalize(lightSource->getPosition() - glm::vec4(intersection.getIntersectionPoint() , 1.0f)), intersection) + color;

}

DEVICE glm::vec4 DRayTracer::findDiffuseColor(DLightSource* lightSource, const glm::vec4& intersectionToLight, DRayIntersection& intersection){
	
	const DMaterial& material = intersection.getIntersectionMaterial();
	float dot = glm::dot(intersectionToLight, glm::vec4(intersection.getIntersectionNormal(), 0.0f));
	dot = glm::max(0.0f, dot);
	return glm::vec4( dot * material.getDiffuseColor() * lightSource->getColor());
}