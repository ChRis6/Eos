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
#include <iostream>

bool Scene::addSurface(Surface* surface){
	if( surface != NULL){
		m_SurfaceObjects.push_back(surface);
		return true;
	}
	return false;
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

const Surface* Scene::getSurface(unsigned int id) const{
	if( id < m_SurfaceObjects.size() )
		return m_SurfaceObjects[id];
	return NULL;
}

const LightSource* Scene::getLightSource( unsigned int id) const{
	if( id < m_LightSources.size())
		return m_LightSources[id];
	return NULL;
}



void Scene::render(const Camera& camera, unsigned char* outputImage){
	
	int viewWidth;
	int viewHeight;
	glm::vec3 viewDirection;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 rayOrigin;	
	//float tan_fovx;
	//float tan_fovy;

	/*
	 * for every pixel on the window
	 *     calculate primary ray
     *	   color = rayTrace(primaryRay)
     *     write color to pixel
     */

    viewWidth  = camera.getWidth();
    viewHeight = camera.getHeight();

    viewDirection = camera.getViewingDirection();
    right         = camera.getRightVector();
    up            = camera.getUpVector();
    rayOrigin     = camera.getPosition();

    for( int y = 0 ; y < viewHeight; y++){
    	// how much 'up'
    	float bb = ((y - (viewHeight/2.0f)) / (float)(viewHeight/ 2.0f));
    	for( int x = 0 ; x < viewWidth; x++){
    		
    		// how much 'right'
    		float aa = (( x - viewWidth/2.0f) / (float)( viewWidth / 2.0f));
			glm::vec3 rayDirection = (aa * right ) + ( bb * up ) + viewDirection;
    		rayDirection = glm::normalize(rayDirection);
    		
    		Ray ray(rayOrigin,rayDirection);
    		// find color
    		glm::vec4 finalColor = this->rayTrace(ray);
    		// store color
    		outputImage[4 * (x + y * viewWidth)]      = floor(finalColor.x == 1.0 ? 255 : std::min(finalColor.x * 256.0, 255.0));
            outputImage[1 +  4 * (x + y * viewWidth)] = floor(finalColor.y == 1.0 ? 255 : std::min(finalColor.y * 256.0, 255.0));
            outputImage[2 +  4* (x + y * viewWidth)]  = floor(finalColor.z == 1.0 ? 255 : std::min(finalColor.z * 256.0, 255.0));
            outputImage[3 +  4* (x + y * viewWidth)]  = 0;
    	}
    }
}

glm::vec4 Scene::rayTrace(const Ray& ray){

	bool rayCollided = false;
	RayIntersection intersection;
	glm::vec4 finalColor(0.0f);
	
	/*
	 * Find min distance intersection
	 */
	rayCollided = this->findMinDistanceIntersection(ray, intersection);

	/*
	 * Apply light
	 */

	// phong
	if(rayCollided == true){

		// find diffuse color for every light source
		int numLights = this->getNumLightSources();
		glm::vec4 diffuseColor(0.0f);
		for( int i = 0 ; i < numLights; i++){
			const LightSource& lightSource = *(m_LightSources[i]);
			diffuseColor += this->findDiffuseColor(lightSource, intersection);
		}
		// add ambient color
		diffuseColor += intersection.getMaterial().getAmbientIntensity() * intersection.getMaterial().getDiffuseColor();
		return diffuseColor;
		
	}
	else{
		// return background color
		return glm::vec4(0.0f);
	}

	return glm::vec4(1.0f);
}

glm::vec4 Scene::findDiffuseColor(const LightSource& lightSource, const RayIntersection& intersection){

	glm::vec4 diffuseColor;
	glm::vec4 intersectionToLight;
	const Material& material = intersection.getMaterial();



	intersectionToLight = glm::normalize(lightSource.getPosition() - glm::vec4(intersection.getPoint(), 1.0f));
	float dot = glm::dot(intersectionToLight, glm::vec4(intersection.getNormal(), 0.0f));
	if( dot > 0.0f){
		return glm::vec4( dot * material.getDiffuseColor() * lightSource.getLightColor());
	}

	// normal is pointing the other way.No color contribution for this light source
	return glm::vec4(0.0f);
}

bool Scene::findMinDistanceIntersection(const Ray& ray, RayIntersection& intersection){

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

		// Transform the ray to local coordinates
		glm::vec3 localRayOrigin    = glm::vec3(glm::inverse(M) * glm::vec4(ray.getOrigin(), 1.0f));
		glm::vec3 localRayDirection = glm::vec3(glm::inverse(M) * glm::vec4(ray.getDirection(), 0.0f)); 
		Ray localRay(localRayOrigin, localRayDirection);

		// check for intersection
		bool intersected = surface.hit(localRay, tempIntersection, minTemp);
		if(intersected){
			intersectionFound = true;
			// is it closer to the ray ?
			if( minTemp < minDist && minTemp > 0.0f){
				// transform intersection point to world coordinates
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