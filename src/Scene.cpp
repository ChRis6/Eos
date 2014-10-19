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
#include <iostream>
#include <omp.h>

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

float Scene::getAmbientRefractiveIndex() const{
	return m_AmbientRefractiveIndex;
}

void Scene::setAmbientRefractiveIndex(float refractiveIndex){
	m_AmbientRefractiveIndex = refractiveIndex;
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

    #pragma omp parallel for schedule(dynamic,1)
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
    		glm::vec4 finalColor = this->rayTrace(ray, camera, this->getAmbientRefractiveIndex(), 0); // 0 depth
    		finalColor = glm::clamp(finalColor, 0.0f, 1.0f);
    		// store color
    		outputImage[4 * (x + y * viewWidth)]      = floor(finalColor.x == 1.0 ? 255 : std::min(finalColor.x * 256.0, 255.0));
            outputImage[1 +  4 * (x + y * viewWidth)] = floor(finalColor.y == 1.0 ? 255 : std::min(finalColor.y * 256.0, 255.0));
            outputImage[2 +  4* (x + y * viewWidth)]  = floor(finalColor.z == 1.0 ? 255 : std::min(finalColor.z * 256.0, 255.0));
            outputImage[3 +  4* (x + y * viewWidth)]  = 255;
            
    	}
    }
}

glm::vec4 Scene::rayTrace(const Ray& ray, const Camera& camera, float sourceRefactionIndex, int depth){

	bool rayCollided = false;
	RayIntersection intersection;
	glm::vec4 finalColor(0.0f);
	
	if( depth > this->getMaxTracedDepth())
		return finalColor;

	/*
	 * Find min distance intersection
	 */
	if(this->isUsingBvh())
		rayCollided = this->findMinDistanceIntersectionBVH(ray, intersection); 		// use bounding volume hierarchy
	else
		rayCollided = this->findMinDistanceIntersectionLinear(ray, intersection);   // use linear search

	/*
	 * Apply light
	 */

	// phong
	if(rayCollided == true){
		finalColor += shadeIntersection(intersection, ray, camera, sourceRefactionIndex, depth);
		return finalColor;
	}
	else{
		// return background color
		return glm::vec4(0.0f);
	}

	return glm::vec4(1.0f);
}

glm::vec4 Scene::shadeIntersection(RayIntersection& intersection, const Ray& ray, const Camera& camera, float sourceRefactionIndex, int depth){

	int numLights;
	const float epsilon = 1e-3f;
	glm::vec4 calculatedColour(0.0f);
	glm::vec3 zeroVector(0.0f);
	RayIntersection dummyIntersection;
	glm::vec4 reflectedColour(0.0f);
	glm::vec4 refractedColour(0.0f);
	glm::vec4 phongColour(0.0f);
	float n1,n2;

	numLights = this->getNumLightSources();
	for( int i = 0 ; i < numLights; i++){
		// light source i
		const LightSource& lightSource = *(m_LightSources[i]);

		
		// find the shadow ray
		glm::vec3 shadowRayDirection = glm::normalize(glm::vec3(lightSource.getPosition()) - intersection.getPoint()); // lightPos - intersection point
		glm::vec3 shadowRayOrigin    = intersection.getPoint() + epsilon * shadowRayDirection;

		Ray shadowRay(shadowRayOrigin, shadowRayDirection);  
		
		bool inShadow = false;
		if( this->isUsingBvh())
			inShadow = this->findMinDistanceIntersectionBVH(shadowRay, dummyIntersection);
		else
			inShadow = this->findMinDistanceIntersectionLinear(shadowRay, dummyIntersection);

		// is the point in shadow ?
		if( inShadow == false) {
			// point is not in shadow.Calculate phong
			phongColour += this->calcPhong(camera, lightSource, intersection);
		}

	}

	// calculate reflections for the surface
	if( intersection.getMaterial().isReflective()){

		glm::vec3 reflectedRayDirection = glm::normalize(glm::reflect(ray.getDirection(), intersection.getNormal()));
		glm::vec3 reflectedRayOrigin    = intersection.getPoint() + epsilon * reflectedRayDirection;

		Ray reflectedRay(reflectedRayOrigin, reflectedRayDirection);

		reflectedColour += intersection.getMaterial().getReflectiveIntensity() * intersection.getMaterial().getSpecularColor() * 
		                   this->rayTrace( reflectedRay, camera, sourceRefactionIndex, depth + 1);
	}
	// is the surface transparent ? 
	if( intersection.getMaterial().isTransparent()){
		
		glm::vec3 zeroVector(0.0f);
		float n1,n2;
		glm::vec3 incident = glm::normalize(ray.getDirection());
		float iDOTn = glm::dot(incident, intersection.getNormal());
		float refractionRatio;
		glm::vec4 tempReflectedColor(0.0f);
		glm::vec4 tempRefractedColor(0.0f);

		glm::vec3 newNormal = intersection.getNormal();

		if( iDOTn < 0.0f ){
			//Ray is outside of the material going in
			n1 = this->getAmbientRefractiveIndex();
			n2 = intersection.getMaterial().getRefractiveIndex();
			//std::cout << "Intersection normal unchanged" << std::endl;
		}
		else{
			// ray is in the material going out
			n1 = sourceRefactionIndex;
			n2 = this->getAmbientRefractiveIndex();
			
			// reverse normal
			//std::cout << "Reversing Intersection normal" << std::endl;
			newNormal = zeroVector - intersection.getNormal();
			
		}
		//std::cout << " n1 = " << n1 << " n2 = " << n2 << std::endl;
		
		refractionRatio = n1 / n2;
		float K = this->fresnel(incident, newNormal, n1, n2);
		
		glm::vec3 reflectedRayDirection = glm::normalize(glm::reflect(ray.getDirection(), newNormal));
		glm::vec3 reflectedRayOrigin    = intersection.getPoint() + epsilon * newNormal;

		Ray reflectedRay(reflectedRayOrigin, reflectedRayDirection);
		tempReflectedColor += this->rayTrace( reflectedRay, camera, n2, depth + 1);


		glm::vec3 refractedRayDirection = glm::refract(incident, newNormal, refractionRatio);
		glm::vec3 refractedRayOrigin    = intersection.getPoint() + epsilon * refractedRayDirection;

		//std::cout << "refracted dot normal" << glm::dot(refractedRayDirection,newNormal) << std::endl;

		Ray refractedRay(refractedRayOrigin, refractedRayDirection);
		tempRefractedColor += this->rayTrace( refractedRay, camera, n2, depth + 1);

		// mix colors
		//K = 0.0f;
		refractedColour += K * tempReflectedColor + (1.0f - K) * tempRefractedColor;
	}


	calculatedColour += phongColour + reflectedColour + refractedColour;

	// add ambient color
	//calculatedColour += intersection.getMaterial().getAmbientIntensity() * intersection.getMaterial().getDiffuseColor();
	return calculatedColour;

}

glm::vec4 Scene::findDiffuseColor(const LightSource& lightSource, RayIntersection& intersection){

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

glm::vec4 Scene::calcPhong( const Camera& camera, const LightSource& lightSource, RayIntersection& intersection){


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
bool Scene::findMinDistanceIntersectionBVH(const Ray& ray, RayIntersection& intersection){

	Surface* intersectedSurface = this->m_Bvh.intersectRay(ray, intersection);
	if( intersectedSurface != NULL)
		return true;
	return false;

}

bool Scene::findMinDistanceIntersectionLinear(const Ray& ray, RayIntersection& intersection){

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

float Scene::fresnel(const glm::vec3& incident, const glm::vec3& normal, float n1, float n2){
	// (Bram de Greve 2006)
	float n = n1 / n2;
	float cosI  = -glm::dot(normal, incident);
	float sinT2 = n * n * ( 1.0f - cosI * cosI);
	if( sinT2 > 1.0f) return 1.0f; // TIR

	float cosT = sqrtf( 1.0f - sinT2);
	float rOrth = ( n1 * cosI - n2 * cosT) / ( n1 * cosI + n2 * cosT);
	float rPar  = (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT);

	return ( rOrth * rOrth + rPar * rPar) / 2.0f; 
}
float Scene::slick(const glm::vec3& incident, const glm::vec3& normal, float n1, float n2){
	float r0 = ( n1 - n2) / ( n1 + n2);
	r0 *= r0;

	float cosX = -glm::dot(normal,incident);
	float cosI = cosX;
	if( n1 > n2){
		float n = n1 / n2;
		float sinT2 = n * n * ( 1.0f - cosI * cosI);
		if( sinT2 >= 1.0f) return 1.0f; // TIR
		cosX = glm::sqrt(1.0f - sinT2);
	} 
	float x = 1.0f - cosX;
	return r0 + ( 1.0f - r0) * x * x * x * x * x;
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