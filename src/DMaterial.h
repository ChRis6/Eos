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

#ifndef _DMATERIAL_H
#define _DMATERIAL_H

#include "cudaQualifiers.h"
#include <glm/glm.hpp>
 
class DMaterial{

public:
	DEVICE DMaterial():m_Diffuse(1.0f),m_Specular(0.0f),m_Ambient(0.015f),m_Reflectivity(0.0f),m_shininess(40){}
	DEVICE DMaterial(const glm::vec4& diffuse, const glm::vec4& specular, const glm::vec4& ambient, float reflectivity, int shininess):
				m_Diffuse(diffuse),m_Specular(specular),m_Ambient(ambient),m_Reflectivity(reflectivity),m_shininess(shininess){}

	DEVICE const glm::vec4& getDiffuseColor() 	{ return m_Diffuse;     }
	DEVICE const glm::vec4& getSpecularColor()  { return m_Specular;    }
	DEVICE const glm::vec4& getAmbientColor()   { return m_Ambient;     }
	DEVICE float getReflectivity()				{ return m_Reflectivity;}
	DEVICE int getShininess()					{ return m_shininess;   }

	DEVICE void setDiffuseColor(const glm::vec4& diffuse) 	{ m_Diffuse  = diffuse; }
	DEVICE void setSpecularColor(const glm::vec4& specular) { m_Specular = specular;}
	DEVICE void setAmbientColor(const glm::vec4& ambient)   { m_Ambient  = ambient; }
	DEVICE void setReflectivity(float reflectivity)			{ m_Reflectivity = reflectivity;}
	DEVICE void setShininess(int shine)						{ m_shininess = shine;  }

private:
	glm::vec4 m_Diffuse;
	glm::vec4 m_Specular;
	glm::vec4 m_Ambient;
	float m_Reflectivity;
	int m_shininess;
};

#endif