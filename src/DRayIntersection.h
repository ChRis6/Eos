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

#ifndef _DRAYINTERSECTION_H
#define _DRAYINTERSECTION_H

#include "cudaQualifiers.h"
#include "DMaterial.h"
#include <glm/glm.hpp>

// thread i has intersection point:Points[i] etc...
typedef struct cudaIntersection{
	
	glm::vec4* points;		// array to intersection points
	glm::vec4* normals;     
	int* materialsIndices;

}cudaIntersection_t;

class DRayIntersection{
public:
	DEVICE DRayIntersection():m_Point(0.0f),m_Normal(0.0f){}
	DEVICE DRayIntersection(const glm::vec4& point, const glm::vec4& normal):
							m_Point(point),m_Normal(normal){}

	DEVICE FORCE_INLINE const glm::vec4& getIntersectionPoint()    { return m_Point;        }
	DEVICE FORCE_INLINE const glm::vec4& getIntersectionNormal()   { return m_Normal;       }
	DEVICE FORCE_INLINE int getIntersectionMaterialIndex()		   { return m_MaterialIndex;}

	DEVICE FORCE_INLINE void setIntersectionPoint(const glm::vec4& point)	 { m_Point    = point;     }
	DEVICE FORCE_INLINE void setIntersectionNormal(const glm::vec4& normal)  { m_Normal   = normal;    }
	DEVICE FORCE_INLINE void setIntersectionMaterialIndex(int index)		 { m_MaterialIndex = index;}
private:
	glm::vec4 m_Point;
	glm::vec4 m_Normal;
	int m_MaterialIndex;
};

#endif