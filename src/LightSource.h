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
#ifndef _LIGHTSOURCE_H
#define _LIGHTSOURCE_H

#include <glm/glm.hpp>

class LightSource{
public:
	LightSource():m_Position(0.0f), m_LightColor(1.0f){}
	LightSource(const glm::vec4& position, const glm::vec4& lightColor):m_Position(position), m_LightColor(lightColor){}

	const glm::vec4& getPosition()   const {return m_Position;}
	const glm::vec4& getLightColor() const {return m_LightColor;}

private:
	glm::vec4 m_Position;
	glm::vec4 m_LightColor;

};

#endif