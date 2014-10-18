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

#ifndef _TEXTURE_H
#define _TEXTURE_H

#include <glm/glm.hpp>

class Texture{

public:
	Texture():m_Data(0),m_Width(0),m_Height(0),m_Channels(0){}
	Texture(char* filename);
	~Texture();
	glm::vec3 getColor(float u, float v);
	char* getPixels() { return m_Data;}
	int getWidth()    { return m_Width;}
	int getHeight()   { return m_Height;}


private:
	char* m_Data; 	// texture data
	int m_Width;		// texture width
	int m_Height;     // texture height
	char m_Channels;  //3 = RGB, 4 = RGBA
};

#endif