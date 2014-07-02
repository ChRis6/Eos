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

#ifndef _MATERIAL_H
#define _MATERIAL_H

#include <glm/glm.hpp>

extern const float REFRACTIVE_INDEX_VACUUM;
extern const float REFRACTIVE_INDEX_AIR;
extern const float REFRACTIVE_INDEX_LIQUID_HELIUM ;
extern const float REFRACTIVE_INDEX_ICE;          
extern const float REFRACTIVE_INDEX_WATER;
extern const float REFRACTIVE_INDEX_CORNEA ;
extern const float REFRACTIVE_INDEX_GLASS_IMPURE;
extern const float REFRACTIVE_INDEX_GLASS_PURE;
extern const float REFRACTIVE_INDEX_DIAMOND;

class Material{

public:
	Material(): m_AmbientIntensity(0.2f), m_DiffuseColor(1.0f), m_SpecularColor(1.0f), m_Shininess(128){}
	Material(float ambientFactor, glm::vec4 diffuseColor, glm::vec4 specularColor, int shininess):
		m_AmbientIntensity(ambientFactor), m_DiffuseColor(diffuseColor), m_SpecularColor(specularColor), m_Shininess(shininess)
		{}

	void setAmbientIntensity(float ambientIntensity) { m_AmbientIntensity    = ambientIntensity;}
	void setDiffuseColor(glm::vec4 diffuseColor)     { m_DiffuseColor        = diffuseColor;}
	void setSpecularColor(glm::vec4 specularColor)   { m_SpecularColor       = specularColor;}
	void setReflective(bool reflective)				 { m_Reflective          = reflective;}
	void setReflectionIntensity(float intensity)     { m_ReflectionIntensity = intensity; }
	void setTransparent(bool transparent)			 { m_Transparent         = transparent;}
	void setRefractiveIndex(float refractiveIndex)   { m_RefractiveIndex     = refractiveIndex;}

	float getAmbientIntensity()    const { return m_AmbientIntensity;}
	glm::vec4 getDiffuseColor()    const { return m_DiffuseColor;}
	glm::vec4 getSpecularColor()   const { return m_SpecularColor;}
	int getShininess()             const { return m_Shininess;}
	bool isReflective()			   const { return m_Reflective;}
	float getReflectiveIntensity() const { return m_ReflectionIntensity;}
	bool isTransparent()		   const { return m_Transparent;}
	float getRefractiveIndex()     const { return m_RefractiveIndex;}

private:
	float m_AmbientIntensity;
	glm::vec4 m_DiffuseColor;
	glm::vec4 m_SpecularColor;
	int m_Shininess;
	bool m_Reflective;
	float m_ReflectionIntensity;

	bool m_Transparent;
	float m_RefractiveIndex;
};

#endif