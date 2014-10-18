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
#include "Texture.h"
#include "stb_image.h"


Texture::Texture(char* filename){
	FILE* file;
	int comp;
	int x,y;
	char* pixels = NULL;

	file = fopen(filename, "rb");
	if(!file){
		fprintf(stderr, "Texture %s not Found\n", filename );
	}
	pixels = (char*) stbi_load_from_file(file, &x, &y, &comp, 0);

	if(pixels == NULL){
		fprintf(stderr, "Texture: %s not Loaded\n", filename );
		exit(1);
	}

	m_Width    = x;
	m_Height   = y;
	m_Channels = comp;
	m_Data     = pixels;
	fclose(file); 
}

glm::vec3 Texture::getColor(float u, float v){
	glm::vec3 color(1.0f);
	float inv = 1 / 255.0f;
	if(m_Data){
		u -= (int)u;
        v -= (int)v;
        
        if(u < 0.0f) u += 1.0f;
        if(v < 0.0f) v += 1.0f;

        int x = (int)(u * m_Width), y = (int)(v * m_Height);

        char *data = (m_Width * y + x) * m_Channels + m_Data;

        color.x = inv * data[0];
        color.y = inv * data[1];
        color.z = inv * data[2];
	}

	return color;
}

Texture::~Texture(){
	if(m_Data)
		delete m_Data;
}