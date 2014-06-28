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
 #include <omp.h>

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

    float norm_width  = viewWidth  / 2.0f;
    float norm_height = viewHeight / 2.0f;  

    float inv_norm_width  = 1.0f / norm_width;
    float inv_norm_height = 1.0f / norm_height; 

    viewDirection = camera.getViewingDirection();
    right         = camera.getRightVector();
    up            = camera.getUpVector();
    rayOrigin     = camera.getPosition();

    #pragma omp parallel for
    for( int y = 0 ; y < viewHeight; y++){
    	// how much 'up'
    	float bb = (y - norm_height) * inv_norm_height;
    	for( int x = 0 ; x < viewWidth; x++){
    		
    		// how much 'right'
    		float aa = ( x - norm_width) * inv_norm_width;
			glm::vec3 rayDirection = (aa * right ) + ( bb * up ) + viewDirection;
    		rayDirection = glm::normalize(rayDirection);
    		
    		Ray ray(rayOrigin,rayDirection);
    		// find color
    		glm::vec4 finalColor = this->rayTrace(ray, camera, 0); // 0 depth
    		// store color
    		outputImage[4 * (x + y * viewWidth)]      = floor(finalColor.x == 1.0 ? 255 : std::min(finalColor.x * 256.0, 255.0));
            outputImage[1 +  4 * (x + y * viewWidth)] = floor(finalColor.y == 1.0 ? 255 : std::min(finalColor.y * 256.0, 255.0));
            outputImage[2 +  4* (x + y * viewWidth)]  = floor(finalColor.z == 1.0 ? 255 : std::min(finalColor.z * 256.0, 255.0));
            outputImage[3 +  4* (x + y * viewWidth)]  = 0;
    	}
    }
}

glm::vec4 Scene::rayTrace(const Ray& ray, const Camera& camera, int depth){

	bool rayCollided = false;
	RayIntersection intersection;
	glm::vec4 finalColor(0.0f);
	
	if( depth > this->getMaxTracedDepth())
		return finalColor;

	/*
	 * Find min distance intersection
	 */
	rayCollided = this->findMinDistanceIntersection(ray, intersection);

	/*
	 * Apply light
	 */

	// phong
	if(rayCollided == true){
		finalColor += shadeIntersection(intersection, ray, camera, depth);
		return finalColor;
	}
	else{
		// return background color
		return glm::vec4(0.0f);
	}

	return glm::vec4(1.0f);
}

glm::vec4 Scene::shadeIntersection(const RayIntersection& intersection, const Ray& ray, const Camera& camera, int depth){

	int numLights;
	const float epsilon = 1e-5;
	glm::vec4 calculatedColour(0.0f);
	RayIntersection dummyIntersection;
	//glm::vec3 epsVector(epsilon, epsilon, epsilon);

	numLights = this->getNumLightSources();
	for( int i = 0 ; i < numLights; i++){
		// light source i
		const LightSource& lightSource = *(m_LightSources[i]);


		// find the shadow ray
		glm::vec3 shadowRayDirection = glm::normalize(glm::vec3(lightSource.getPosition()) - intersection.getPoint()); // lightPos - intersection point
		glm::vec3 shadowRayOrigin    = intersection.getPoint() + epsilon * shadowRayDirection;

		Ray shadowRay(shadowRayOrigin, shadowRayDirection);  
		
		// is the point in shadow ?
		if( this->findMinDistanceIntersection(shadowRay, dummyIntersection) == false) {
			// point is not in shadow.Calculate phong
			calculatedColour += this->calcPhong(camera, lightSource, intersection);
		}
	}

	if( intersection.getMaterial().isReflective()){

		glm::vec3 reflectedRayDirection = glm::normalize(ray.getDirection() - 2.0f * (glm::dot(ray.getDirection(), intersection.getNormal())) * intersection.getNormal());
		glm::vec3 reflectedRayOrigin    = intersection.getPoint() + epsilon * reflectedRayDirection;

		Ray reflectedRay(reflectedRayOrigin, reflectedRayDirection);

		calculatedColour += intersection.getMaterial().getSpecularColor() * this->rayTrace( reflectedRay, camera, depth + 1);
	}

	// add ambient color
	calculatedColour += intersection.getMaterial().getAmbientIntensity() * intersection.getMaterial().getDiffuseColor();
	return calculatedColour;

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

glm::vec4 Scene::calcPhong( const Camera& camera, const LightSource& lightSource, const RayIntersection& intersection){


	glm::vec4 diffuseColor(0.0f, 0.0f, 0.0f, 0.0f);
	glm::vec4 specularColor(0.0f, 0.0f, 0.0f, 0.0f);
	glm::vec4 intersectionToLight;
	glm::vec4 reflectedVector;
	glm::vec4 viewVector;

	glm::vec4 intersectionPointInWorld  = glm::vec4(intersection.getPoint() , 1.0f);
	glm::vec4 intersectionNormalInWorld = glm::vec4(intersection.getNormal(), 0.0f);

	diffuseColor = this->findDiffuseColor(lightSource, intersection);

	// add specular reflection
	intersectionToLight = glm::normalize(lightSource.getPosition() - intersectionPointInWorld);
	viewVector          = glm::normalize(glm::vec4(camera.getPosition(),1.0f) - intersectionPointInWorld);
	reflectedVector     = glm::normalize((2.0f * glm::dot(intersectionNormalInWorld, intersectionToLight) * intersectionNormalInWorld) - intersectionToLight);

	float dot = glm::dot( viewVector, reflectedVector);
	if( dot > 0.0f){
		float specularTerm = glm::pow(dot, (float)intersection.getMaterial().getShininess());
		specularColor += specularTerm * lightSource.getLightColor() * intersection.getMaterial().getSpecularColor();
	}

	return diffuseColor + specularColor;
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