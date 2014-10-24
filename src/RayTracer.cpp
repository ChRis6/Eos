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

#include <stdio.h>
#include <omp.h>


#include "RayTracer.h"
#include "Ray.h"
#include "RayIntersection.h"

//#define PRINT_PROGRESS

// setters - getters
void RayTracer::setAASamples(int samples){
	if(samples > 0)
		m_AASamples = samples;
}

int RayTracer::getAASamples() const{
	return m_AASamples;
}

int RayTracer::getTracedDepth() const{
	return m_TraceDepth;
}

void RayTracer::setTracedDepth(int depth){
	if(depth > 0)
		m_TraceDepth = depth;
}

float RayTracer::fresnel(const glm::vec3& incident, const glm::vec3& normal, float n1, float n2){
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

float RayTracer::slick(const glm::vec3& incident, const glm::vec3& normal, float n1, float n2){
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




void RayTracer::render(const Scene& scene, const Camera& camera, unsigned char* outputImage){

	int aasamples;


	aasamples = this->getAASamples();
	if(aasamples == RAYTRACER_NO_AA)
		renderWithoutAA(scene, camera, outputImage);
	else
		renderWithAA(scene, camera, outputImage);

}


void RayTracer::renderWithoutAA(const Scene& scene, const Camera& camera, unsigned char* outputImage){
	int viewWidth;
	int viewHeight;
	glm::vec3 viewDirection;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 rayOrigin;	


	int progress = 0;
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


    #pragma omp parallel for schedule(dynamic,2) shared(progress)
    for( int y = 0 ; y < viewHeight; y++){
    	Ray ray;
    	// how much 'up'
    	float bb = (y - norm_height) * inv_norm_height;  
    	for( int x = 0 ; x < viewWidth; x++){
    		
    		// how much 'right'
    		float aa = ( x - norm_width) * inv_norm_width;

			glm::vec3 rayDirection = (aa * right ) + ( bb * up ) + viewDirection;
    		rayDirection = glm::normalize(rayDirection);
    		
    		ray.setOrigin(rayOrigin);
    		ray.setDirection(rayDirection);
    		// find color
    		glm::vec4 finalColor = this->rayTrace(scene, camera, ray, 0);

    		// store color
    		outputImage[4 * (x + y * viewWidth)]      = floor(finalColor.x == 1.0 ? 255 : std::min(finalColor.x * 256.0, 255.0));
            outputImage[1 +  4 * (x + y * viewWidth)] = floor(finalColor.y == 1.0 ? 255 : std::min(finalColor.y * 256.0, 255.0));
            outputImage[2 +  4* (x + y * viewWidth)]  = floor(finalColor.z == 1.0 ? 255 : std::min(finalColor.z * 256.0, 255.0));
            outputImage[3 +  4* (x + y * viewWidth)]  = 255;
            
    	}
    	#ifdef PRINT_PROGRESS
    		#pragma omp atomic
    		progress++;

    		#pragma omp critical 
    		{
    			if( omp_get_thread_num() == 0 ){
    				printf("\rRendering Progress:%d%%", (int)(( progress / (float) viewHeight ) * 100));
    				fflush(stdout);
    			}
    		}
    	#endif
    }
    #ifdef PRINT_PROGRESS
    	printf("\n");
    #endif


}

void RayTracer::renderWithAA(const Scene& scene, const Camera& camera, unsigned char* outputImage){
	

	int viewWidth;
	int viewHeight;
	glm::vec3 viewDirection;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 rayOrigin;	


	int progress = 0;
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

    int aaSamples = this->getAASamples();
    float ksi = rand() / RAND_MAX;
    float inv_aaSamples = 1.0 / (float) aaSamples;
    float inv_aaSamplesSquared = inv_aaSamples * inv_aaSamples;

    #pragma omp parallel for schedule(dynamic,2) shared(progress)
    for( int y = 0 ; y < viewHeight; y++){
    	Ray ray;
    	for( int x = 0 ; x < viewWidth; x++){
    		
    		glm::vec4 finalColor(0.0f);

    		for( int q = 0 ; q < aaSamples ; q++){
    			// how much 'up'
    			float bb = (y + (( q + ksi ) * inv_aaSamples) - norm_height) * inv_norm_height;    			
    			for( int p = 0; p < aaSamples; p++){
    				// how much 'right'
    				float aa = ( x + (( p + ksi ) * inv_aaSamples)- norm_width) * inv_norm_width;

					glm::vec3 rayDirection = (aa * right ) + ( bb * up ) + viewDirection;
    				rayDirection = glm::normalize(rayDirection);
    		
    				ray.setOrigin(rayOrigin);
    				ray.setDirection(rayDirection);
    				// find color
    				finalColor += this->rayTrace(scene, camera, ray, 0);
    			}
    		}

    		finalColor = finalColor * inv_aaSamplesSquared;
    		
    		// store color
    		outputImage[4 * (x + y * viewWidth)]      = floor(finalColor.x == 1.0 ? 255 : std::min(finalColor.x * 256.0, 255.0));
            outputImage[1 +  4 * (x + y * viewWidth)] = floor(finalColor.y == 1.0 ? 255 : std::min(finalColor.y * 256.0, 255.0));
            outputImage[2 +  4* (x + y * viewWidth)]  = floor(finalColor.z == 1.0 ? 255 : std::min(finalColor.z * 256.0, 255.0));
            outputImage[3 +  4* (x + y * viewWidth)]  = 255;
            
    	}
    	#ifdef PRINT_PROGRESS
    		#pragma omp atomic
    		progress++;

    		#pragma omp critical 
    		{
    			if( omp_get_thread_num() == 0 ){
    				printf("\rRendering Progress:%d%%", (int)(( progress / (float) viewHeight ) * 100));
    				fflush(stdout);
    			}
    		}
    	#endif
    }
    #ifdef PRINT_PROGRESS
    	printf("\n");
    #endif
}


glm::vec4 RayTracer::rayTrace(const Scene& scene, const Camera& camera, const Ray& ray, int depth){

	bool rayCollided = false;
	RayIntersection intersection;
	glm::vec4 finalColor(0.0f);
	
	if( depth > this->getTracedDepth())
		return finalColor;

	/*
	 * Find min distance intersection
	 */
	if(scene.isUsingBvh())
		rayCollided = scene.findMinDistanceIntersectionBVH(ray, intersection); 		// use bounding volume hierarchy
	else
		rayCollided = scene.findMinDistanceIntersectionLinear(ray, intersection);   // use linear search

	/*
	 * Apply light
	 */

	// phong
	if(rayCollided == true){
		finalColor += shadeIntersection(scene, ray, camera, intersection, depth);
		return finalColor;
	}
	else{
		// return background color
		return glm::vec4(0.1f);
	}

}

glm::vec4 RayTracer::shadeIntersection(const Scene& scene, const Ray& ray, const Camera& camera, RayIntersection& intersection, int depth){

	int numLights;
	const float epsilon = 1e-3f;
	glm::vec4 calculatedColour(0.0f);
	glm::vec3 zeroVector(0.0f);
	RayIntersection dummyIntersection;
	glm::vec4 reflectedColour(0.0f);
	glm::vec4 refractedColour(0.0f);
	glm::vec4 phongColour(0.0f);
	float n1,n2;
	
	Ray tempRay;	// use one ray for shadows - reflections

	const glm::vec3& intersectionPoint = intersection.getPoint();
	numLights = scene.getNumLightSources();
	for( int i = 0 ; i < numLights; i++){
		// light source i
		const LightSource* lightSource = scene.getLightSource(i);

		
		// find the shadow ray
		glm::vec3 shadowRayDirection = glm::normalize(glm::vec3(lightSource->getPosition()) - intersectionPoint); // lightPos - intersection point
		glm::vec3 shadowRayOrigin    = intersectionPoint + epsilon * shadowRayDirection;

		//Ray shadowRay(shadowRayOrigin, shadowRayDirection);
		tempRay.setOrigin(shadowRayOrigin);
		tempRay.setDirection(shadowRayDirection);  
		
		bool inShadow = false;
		if( scene.isUsingBvh())
			inShadow = scene.shadowRayVisibilityBVH(tempRay);
		else
			inShadow = scene.findMinDistanceIntersectionLinear(tempRay, dummyIntersection);

		// is the point in shadow ?
		if( inShadow == false) {
			// point is not in shadow.Calculate phong
			phongColour += this->calcPhong(camera, lightSource, intersection);
		}

	}

	// calculate reflections for the surface
	if( intersection.getMaterial().isReflective()){
		
		glm::vec3 reflectedRayDirection = glm::normalize(glm::reflect(ray.getDirection(), intersection.getNormal()));
		glm::vec3 reflectedRayOrigin    = intersectionPoint + epsilon * reflectedRayDirection;

		
		tempRay.setOrigin(reflectedRayOrigin);
		tempRay.setDirection(reflectedRayDirection);
		reflectedColour += intersection.getMaterial().getReflectiveIntensity() * 
		                   this->rayTrace(scene, camera, tempRay, depth + 1);
		
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
			n1 = scene.getAmbientRefractiveIndex();
			n2 = intersection.getMaterial().getRefractiveIndex();
		}
		else{
			// ray is in the material going out
			n1 = intersection.getMaterial().getRefractiveIndex();
			n2 = scene.getAmbientRefractiveIndex();
			
			// reverse normal
			newNormal = zeroVector - intersection.getNormal();
			
		}
		
		refractionRatio = n1 / n2;
		float K = this->fresnel(incident, newNormal, n1, n2);
		
		glm::vec3 reflectedRayDirection = glm::normalize(glm::reflect(ray.getDirection(), newNormal));
		glm::vec3 reflectedRayOrigin    = intersectionPoint + epsilon * newNormal;

		Ray reflectedRay(reflectedRayOrigin, reflectedRayDirection);
		tempReflectedColor += this->rayTrace( scene, camera, reflectedRay, depth + 1);


		glm::vec3 refractedRayDirection = glm::refract(incident, newNormal, refractionRatio);
		glm::vec3 refractedRayOrigin    = intersectionPoint + epsilon * refractedRayDirection;


		Ray refractedRay(refractedRayOrigin, refractedRayDirection);
		tempRefractedColor += this->rayTrace( scene, camera, reflectedRay, depth + 1);

		// mix colors
		refractedColour += K * tempReflectedColor + (1.0f - K) * tempRefractedColor;
	}


	calculatedColour += phongColour + reflectedColour + refractedColour;

	// add ambient color
	//calculatedColour += intersection.getMaterial().getAmbientIntensity() * intersection.getMaterial().getDiffuseColor();
	return calculatedColour;

}

glm::vec4 RayTracer::calcPhong( const Camera& camera, const LightSource* lightSource, RayIntersection& intersection){

	glm::vec4 diffuseColor(0.0f, 0.0f, 0.0f, 0.0f);
	glm::vec4 specularColor(0.0f, 0.0f, 0.0f, 0.0f);
	glm::vec4 intersectionToLight;
	glm::vec4 reflectedVector;
	glm::vec4 viewVector;
	
	glm::vec4 intersectionPointInWorld  = glm::vec4(intersection.getPoint() , 1.0f);
	glm::vec4 intersectionNormalInWorld = glm::vec4(intersection.getNormal(), 0.0f);

	
	// specular reflection
	intersectionToLight = glm::normalize(lightSource->getPosition() - intersectionPointInWorld);
	viewVector          = glm::normalize(glm::vec4(camera.getPosition(),1.0f) - intersectionPointInWorld);
	reflectedVector     = glm::normalize((2.0f * glm::dot(intersectionNormalInWorld, intersectionToLight) * intersectionNormalInWorld) - intersectionToLight);
	
	// find diffuse first
	diffuseColor = this->findDiffuseColor(lightSource, intersectionToLight, intersection);

	float dot = glm::dot( viewVector, reflectedVector);
	if( dot > 0.0f){
		float specularTerm = glm::pow(dot, (float)intersection.getMaterial().getShininess());
		specularColor += specularTerm * lightSource->getLightColor() * intersection.getMaterial().getSpecularColor();
	}

	return diffuseColor + specularColor;
}

glm::vec4 RayTracer::findDiffuseColor(const LightSource* lightSource, const glm::vec4& intersectionToLight, const RayIntersection& intersection){
	glm::vec4 diffuseColor;
	const Material& material = intersection.getMaterial();


	float dot = glm::dot(intersectionToLight, glm::vec4(intersection.getNormal(), 0.0f));
	dot = glm::max(0.0f, dot);
	return glm::vec4( dot * material.getDiffuseColor() * lightSource->getLightColor());
}
