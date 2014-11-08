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

#include "Scene.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

//#define PRINT_PROGRESS


bool Scene::addSurface(Surface* surface){
	if( surface != NULL){
		m_SurfaceObjects.push_back(surface);
		return true;
	}
	return false;
}

bool Scene::addTriangleMesh(TriangleMesh* mesh){
	int i;
	int numTriangleSurfaces;
	int sceneNumObjects;

	if(!mesh)
		return false;

	numTriangleSurfaces = mesh->getNumTriangles();
	sceneNumObjects = this->getNumSurfaces();

	for( i = 0 ; i < numTriangleSurfaces; i++){
		Triangle* tri = mesh->getTriangle(i);
		if(tri != NULL)
			m_SurfaceObjects.push_back(tri);
	}
	mesh->setSceneStartIndex(sceneNumObjects);
	mesh->setSceneEndIndex(this->getNumSurfaces() - sceneNumObjects);
	
	return true;
}

bool Scene::addLightSource(LightSource* light){
	if( light != NULL){
		m_LightSources.push_back(light);
		return true;
	}
	return false;
}

int Scene::getNumSurfaces() const{
	return m_SurfaceObjects.size();
}

int Scene::getNumLightSources() const {
	return m_LightSources.size();
}


Surface* Scene::getSurface( int id) const{
	if( id < m_SurfaceObjects.size() )
		return m_SurfaceObjects[id];
	return NULL;
}

const LightSource* Scene::getLightSource( int id) const{
	if( id < m_LightSources.size())
		return m_LightSources[id];
	return NULL;
}

float Scene::getAmbientRefractiveIndex() const{
	return m_AmbientRefractiveIndex;
}

void Scene::setAmbientRefractiveIndex(float refractiveIndex){
	m_AmbientRefractiveIndex = refractiveIndex;
}








bool Scene::findMinDistanceIntersectionBVH(const Ray& ray, RayIntersection& intersection) const{
	/*
	Surface* intersectedSurface = this->m_Bvh.intersectRay(ray, intersection);
	if( intersectedSurface != NULL)
		return true;
	return false;
	*/
	return this->m_Bvh.intersectRay(ray, intersection, true, this->getFirstSurfaceAddress());
}

bool Scene::shadowRayVisibilityBVH(const Ray& ray) const{
	RayIntersection dummyIntersection;
	return this->m_Bvh.intersectRay(ray, dummyIntersection, false, this->getFirstSurfaceAddress());
}
bool Scene::findMinDistanceIntersectionLinear(const Ray& ray, RayIntersection& intersection) const{

	bool intersectionFound;
	int numSurfaces;
	float minDist = 999999;
	float minTemp;
	glm::vec4 min_point(0.0f);
	glm::vec4 min_normal(0.0f);

	RayIntersection tempIntersection;

	numSurfaces = this->getNumSurfaces();
	intersectionFound = false;

	for( int i = 0; i < numSurfaces; i++){
		Surface& surface = *(m_SurfaceObjects[i]);
		const glm::mat4& M = surface.transformation();

		// Transform the ray to object local coordinates
		glm::vec3 localRayOrigin    = glm::vec3(glm::inverse(M) * glm::vec4(ray.getOrigin(), 1.0f));
		glm::vec3 localRayDirection = glm::vec3(glm::inverse(M) * glm::vec4(ray.getDirection(), 0.0f)); 
		Ray localRay(localRayOrigin, localRayDirection);

		// check for intersection
		bool intersected = surface.hit(localRay, tempIntersection, minTemp);
		if(intersected){
			intersectionFound = true;
			// is it closer to the ray ?
			if( minTemp < minDist && minTemp > 0.0f){ 		// closest positive value
				// transform intersection point to world coordinates
				minDist = minTemp;
				min_point  = M * glm::vec4(tempIntersection.getPoint(), 1.0f);
				min_normal = glm::transpose(glm::inverse(M)) * glm::vec4(tempIntersection.getNormal(), 0.0f);
				intersection.setMaterial(surface.getMaterial()); 
			}
		}
	}

	// transform intersection back to world coordinates;
	intersection.setPoint(glm::vec3(min_point));
	intersection.setNormal(glm::vec3(min_normal));
	return intersectionFound;
}


void Scene::flush(){
	if(this->isUsingBvh() == true){
		// build hierarchy
		m_Bvh.buildHierarchy(&m_SurfaceObjects[0], this->getNumSurfaces());
	}
}

glm::vec3 Scene::getReflectedRay(const glm::vec3& rayDir, const glm::vec3& normal){
	float c = -glm::dot(rayDir, normal);
	glm::vec3 reflectedDirection = rayDir + ( 2.0f * normal * c );
	return reflectedDirection;
}

glm::vec3 Scene::getRefractedRay(const glm::vec3& rayDir, const glm::vec3& normal, float n1, float n2){
	float c1 = -glm::dot(rayDir, normal);
	float ratio = n1 / n2;
	float sinT2 = ratio * ratio * (1.0f - c1 * c1);
	if( sinT2 > 1.0f) return glm::vec3(0.0f);
	float cosT = glm::sqrt(1.0f - sinT2);

	glm::vec3 refractedDirection = ratio * rayDir + ( ratio * c1  - cosT ) * normal;
	return refractedDirection;
}


void Scene::printScene(){

	int numLights;
	int numTriangles;
	int i;
	printf("Priting Scene from Host\n\n");	

	numLights = this->getNumLightSources();
	numTriangles = this->getNumSurfaces();

	printf("Scene has %d lights and %d triangles\n", numLights, numTriangles);

	for( i = 0 ; i < numLights; i++){
		const LightSource* light = this->getLightSource(i);
		glm::vec4 pos = light->getPosition();
		glm::vec4 color = light->getLightColor();

		printf("LightSource: %d\n", i+1);
		printf("Position: %f %f %f\n", pos.x, pos.y, pos.z);
		printf("Color: %f %f %f\n\n", color.x, color.y, color.z);
	}


	for( i = 0 ; i < numTriangles; i++){

		Surface* surface = this->getSurface(i);
		Triangle* tri = dynamic_cast<Triangle*>(surface);
		glm::vec3 v1 = tri->m_V1;
		glm::vec3 v2 = tri->m_V2;
		glm::vec3 v3 = tri->m_V3;


		glm::vec3 n1 = tri->m_N1;
		glm::vec3 n2 = tri->m_N2;
		glm::vec3 n3 = tri->m_N3;

		printf("Triangle: %d\n", i+1);
		printf("V1: %f %f %f\n", v1.x, v1.y, v1.z);
		printf("V2: %f %f %f\n", v2.x, v2.y, v2.z);
		printf("V3: %f %f %f\n\n", v3.x, v3.y, v3.z);

		printf("N1: %f %f %f\n", n1.x, n1.y, n1.z);
		printf("N2: %f %f %f\n", n2.x, n2.y, n2.z);
		printf("N3: %f %f %f\n\n", n3.x, n3.y, n3.z);
	}
}