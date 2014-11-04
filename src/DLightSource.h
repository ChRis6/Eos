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

#ifndef _DLIGHT_SOURCE_H
#define _DLIGHT_SOURCE_H

#include "cudaQualifiers.h"

#include <glm/glm.hpp>
class DeviceSceneHandler;

class DLightSource{
	friend class DeviceSceneHandler;
public: // constructors
	DEVICE DLightSource(const glm::vec4& pos, const glm::vec4& color):m_Position(pos), m_Color(color){}

public: // setters - getters
	DEVICE void setPosition(const glm::vec4 &pos) { m_Position = pos;}
	DEVICE const glm::vec4& getPosition()         { return m_Position;} 

	DEVICE void setColor(const glm::vec4& color)  { m_Color = color;}
	DEVICE const glm::vec4& getColor()            { return m_Color;}

private: // methods used ONLY by DeviceSceneHandler class
	HOST DLightSource():m_Position(0.0f),m_Color(1.0f){}

private:
	glm::vec4 m_Position;
	glm::vec4 m_Color;
};

#endif