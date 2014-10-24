#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "stb_image.h"
#include "Ray.h"
#include "sphere.h"
#include "RayIntersection.h"
//#include "triangleMesh.h"
#include "Camera.h"
#include "Scene.h"
#include "LightSource.h"
#include "Material.h"

#include "cudaWrapper.h"
#include "getTime.h"
#include "Texture.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define WINDOW_WIDTH   640  // in pixels
#define WINDOW_HEIGHT  480 // in pixels
#define FOV            70

#ifndef EPSILON
#define EPSILON        1e-3
#endif

void write_ppm(int width, int height, unsigned char* image){
  
  
  int x, y;
  char* imagechar = (char*) image;
  FILE *fp = fopen("raytracedScene.ppm", "wb"); 
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  unsigned char color[3];

  for (y = height; y > 0; --y)
  {
    for (x = 0; x < width; ++x)
    {
      color[0] = imagechar[0 + 4 * (x + y * width)] ;  
      color[1] = imagechar[1 + 4 * (x + y * width)];  
      color[2] = imagechar[2 + 4 * (x + y * width)];
      fwrite(color, 1, 3, fp);
   }
  }


  fclose(fp);

}

void glfwErrorCallback(int error, const char *description)
{
   std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

std::string readShaderFromFile(const char* filename){

   std::ifstream t(filename);
   std::string str;

   t.seekg(0, std::ios::end);   
   str.reserve(t.tellg());
   t.seekg(0, std::ios::beg);

   str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

   return str;
}


GLuint loadShader( const char* vertexShaderSrc, const  char* fragmentShaderSrc){

 
   // Create the shaders
   GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
   GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

 
   GLint res = GL_FALSE;
   int InfoLogLength;
 
   // Compile Vertex Shader
   glShaderSource(VertexShaderID, 1, &vertexShaderSrc , NULL);
   glCompileShader(VertexShaderID);
 
   // Check Vertex Shader
   glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &res);
   if( res == GL_FALSE){

      char log[1024];

      glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
      glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL , &log[0]);

      std::cout << "Vertex Shader: " << vertexShaderSrc << std::endl;

      std::cout << "Error: " << log << std::cout;
      return 0;
   }
 
    // Compile Fragment Shader
   glShaderSource(FragmentShaderID, 1, &fragmentShaderSrc, NULL);
   glCompileShader(FragmentShaderID);
 
   // Check Fragment Shader
   glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &res);
   if( res == GL_FALSE){

      char log[1024];

      glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
      glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL , &log[0]);

      std::cout << " Fragment Shader: " <<  fragmentShaderSrc << std::endl;

      std::cout << "Error: " << log << std::cout;
      glDeleteShader(VertexShaderID);
      return 0;
   }
 
   // Link the program
   GLuint ProgramID = glCreateProgram();
   if( ProgramID == 0 ){

      glDeleteShader(VertexShaderID);
      glDeleteShader(FragmentShaderID);
      return 0;
   }
   glAttachShader(ProgramID, VertexShaderID);
   glAttachShader(ProgramID, FragmentShaderID);
   glLinkProgram(ProgramID);
 
   // Check status
   glGetProgramiv(ProgramID, GL_LINK_STATUS, &res);
   if( res == GL_FALSE ){

      char log[1024];

      glGetProgramiv( ProgramID , GL_INFO_LOG_LENGTH, &InfoLogLength);
      glGetProgramInfoLog(ProgramID, InfoLogLength, NULL , &log[0]);

      std::cout << "Vertex Shader Source" << vertexShaderSrc << std::endl;
      std::cout << "Fragment Shader Source" << fragmentShaderSrc << std::endl;

      std::cout << "Error linking: " << log << std::endl;

      glDeleteProgram(ProgramID);
      glDeleteShader(VertexShaderID);
      glDeleteShader(FragmentShaderID);

      return 0;
   }

   // detach and delete shaders.Be nice to the driver
   glDetachShader(ProgramID, VertexShaderID);
   glDetachShader(ProgramID, FragmentShaderID);

   glDeleteShader(VertexShaderID);
   glDeleteShader(FragmentShaderID);

   return ProgramID;
}


int main(int argc, char **argv)
{
   
   
   bool renderOnce = false;

   srand((int) time(NULL));


   GLuint vao;
   GLFWwindow* window;
   GLuint program;
   GLuint vbo;
   GLuint pbo;
   GLuint texture;

   if(!renderOnce){
   		glfwSetErrorCallback(glfwErrorCallback);

   		if(!glfwInit())
   		{
      		std::cerr << "Failed to initialize GLFW...\n";
      		return -1;
   		}
	}

   unsigned char* imageBuffer = new unsigned char[ WINDOW_WIDTH * WINDOW_HEIGHT * 4];
   memset(imageBuffer, 0, sizeof(unsigned char) * 4 * WINDOW_WIDTH * WINDOW_HEIGHT);

   if(!renderOnce){
   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

   
   window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Eos Renderer", NULL, NULL);
   if(!window)
   {
      std::cerr << "Failed to open GLFW window...\n";
      glfwTerminate();
      return -1;
   }

   std::cout<< "Creating context\n";
   glfwMakeContextCurrent(window);

   //std::cout << "Going to initialize GLEW...\n";
   glewExperimental = GL_TRUE;
   if (glewInit() != GLEW_OK)
   {
      std::cerr << "Failed to initialize GLEW...\n";
      glfwTerminate();
      return -1;
   }

   // hide the cursor
   glfwSetInputMode (window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
   // vsync = off
   glfwSwapInterval(0);     



   std::cout << "Creating Vertex Array...\n";
   
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   // set up shaders
   std::string vertexShaderSource = readShaderFromFile("shaders/simple.vs");
   std::string fragmentShaderSource = readShaderFromFile("shaders/simple.fs");

   program = loadShader( vertexShaderSource.c_str(), fragmentShaderSource.c_str() );
   if( program == 0 ){
      std::cout << "Program not created...\n";
      glfwTerminate();
      return -1;
   }

   // screen quad
   float quad[] = { -1.0f, -1.0f, 0.0f, // bottom-left
                     -1.0f, 1.0f, 0.0f, // top-left
                     1.0f, -1.0f, 0.0f, // bottom-right

                     -1.0f, 1.0f, 0.0f, // top-left
                      1.0f, 1.0f, 0.0f, // top-right
                      1.0f, -1.0f, 0.0f // bottom-right 
                   };

   
   glGenBuffers(1, &vbo);

   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);   // copy to device memory


   
   glGenBuffers(1, &pbo);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
   glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(GLchar) * WINDOW_WIDTH * WINDOW_HEIGHT * 4, imageBuffer, GL_DYNAMIC_DRAW);


   // generate texture
   
   glGenTextures(1, &texture);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);


   glUseProgram(program);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);


   // transfer pixels to texture via pixel buffer object
   glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH , WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
   // filtering
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  
   glfwSetCursorPos(window, WINDOW_WIDTH/2.0, WINDOW_HEIGHT/2.0);
   }

   

   glm::mat4 M = glm::translate( glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -0.9f));
   
  

   Camera camera;

   camera.setPosition(glm::vec3(2.0f, 0.0f,5.0f));
   camera.setViewingDirection(glm::vec3(0.0f,0.0f, -1.0f));
   camera.setRightVector(glm::vec3(1.0f, 0.0f, 0.0f));
   camera.setUpVector(glm::vec3(0.0f, 1.0f, 0.0f));
   camera.setFov(FOV);
   camera.setWidth(WINDOW_WIDTH);
   camera.setHeight(WINDOW_HEIGHT);

   float horizontalAngle = 0.0f;
   float verticalAngle = -45.0f;
   const float maxAbsoluteVerticalAngle = 90.0f - 0.001f;
   float speed = 5.0f; 
   float mouseSpeed = 0.005f;

/*
   Texture* earthTex = new Texture("earth.png");
   if(!earthTex){
      fprintf(stderr, "Texture not found\n" );
      return -1;
   }
*/


   double currTime = 0.0;
   double timeAcc = 0.0;
   int fps = 0;
   float deltaTime = 0.0f;

   // set up scene
   Scene scene;
   scene.setMaxTracedDepth(5);
   scene.setAmbientRefractiveIndex(REFRACTIVE_INDEX_AIR);
   scene.useBvh(true);
   scene.setAASamples(SCENE_AA_1);


   float refletionIntensity = 0.4f;
   Material sphereMaterial(0.075f, glm::vec4(0.0f, 0.0f, 1.0f, 0.0f), glm::vec4(1.0f), 40);
   sphereMaterial.setReflective(true);
   sphereMaterial.setReflectionIntensity(refletionIntensity);
   sphereMaterial.setTransparent(false);
   sphereMaterial.setRefractiveIndex(REFRACTIVE_INDEX_WATER);
   

   Material sphereMaterial1(0.075f, glm::vec4(0.0f, 1.0f, 0.0f, 0.0f), glm::vec4(1.0f), 120);
   sphereMaterial1.setReflective(true);
   sphereMaterial1.setTransparent(false);
   sphereMaterial1.setReflectionIntensity(refletionIntensity);
   sphereMaterial1.setRefractiveIndex(REFRACTIVE_INDEX_WATER);
 

   Material sphereMaterial2(0.075f, glm::vec4(1.0f, 1.0f, 0.0f, 0.0f), glm::vec4(1.0f), 120);
   sphereMaterial2.setReflective(true);
   sphereMaterial2.setTransparent(false);
   sphereMaterial2.setReflectionIntensity(refletionIntensity);
   

   Material sphereMaterial3(0.075f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f), glm::vec4(1.0f), 120);
   sphereMaterial3.setReflective(true);
   sphereMaterial3.setTransparent(false);
   sphereMaterial3.setReflectionIntensity(refletionIntensity);
   sphereMaterial3.setRefractiveIndex(REFRACTIVE_INDEX_WATER);
   

   Material sphereMaterial4(0.075f, glm::vec4(0.5f, 0.5f, 0.5f, 1.0f), glm::vec4(1.0f), 120);
   sphereMaterial4.setReflective(true);
   sphereMaterial4.setTransparent(false);
   sphereMaterial4.setReflectionIntensity(refletionIntensity);
   sphereMaterial4.setRefractiveIndex(REFRACTIVE_INDEX_AMBER);
   

   Material gridMaterial(0.1f, glm::vec4(0.45f, 0.45f, 0.45f, 0.0f), glm::vec4(0.0f), 140);
   gridMaterial.setTransparent(false);
   gridMaterial.setReflective(true);
   gridMaterial.setReflectionIntensity(refletionIntensity);
   gridMaterial.setRefractiveIndex(REFRACTIVE_INDEX_WATER);

   Material gridMaterialLeft(0.1f, glm::vec4(0.0f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 0.0f), 140);
   gridMaterialLeft.setTransparent(false);
   gridMaterialLeft.setReflective(true);
   gridMaterialLeft.setReflectionIntensity(1.0f); // mirror
   gridMaterialLeft.setRefractiveIndex(REFRACTIVE_INDEX_WATER);

   Material triangleMeshMaterial(0.1f, glm::vec4(0.45f, 0.45f, 0.45f, 0.0f), glm::vec4(1.0f), 120);
   triangleMeshMaterial.setTransparent(false);
   triangleMeshMaterial.setReflective(false);
   triangleMeshMaterial.setReflectionIntensity(refletionIntensity);
   triangleMeshMaterial.setRefractiveIndex(REFRACTIVE_INDEX_WATER);

   LightSource* lightSource  = new LightSource(glm::vec4(-10.0f, 30.0f, -20.0f, 1.0f), glm::vec4(1.0f));  // location , color
   LightSource* lightSource1 = new LightSource(glm::vec4(20.0f, 30.0f, 20.0f, 1.0f), glm::vec4(1.0f));
   //LightSource* lightSource2 = new LightSource(glm::vec4(2000.0f, 0.0f, 40.0f, 1.0f), glm::vec4(1.0f));

   Sphere* sphere = new Sphere(glm::vec3(0.8f, 1.65f, 0.0f), 0.7f);
   sphere->setTransformation(M);
   sphere->setMaterial(sphereMaterial);
  
   scene.addSurface(sphere);

   Sphere* sphere1 = new Sphere(glm::vec3(0.8f, 0.0f, 0.0f), 0.7f);
   sphere1->setTransformation(M);
   sphere1->setMaterial(sphereMaterial1);
   
   scene.addSurface(sphere1);
   
   Sphere* sphere2 = new Sphere(glm::vec3(-0.8f, 0.0f, 0.0f), 0.7f);
   sphere2->setTransformation(M);
   sphere2->setMaterial(sphereMaterial2);
   
   scene.addSurface(sphere2);

   Sphere* sphere3 = new Sphere(glm::vec3(-0.8f, 1.65f, 0.0f), 0.7f);
   sphere3->setTransformation(M);
   sphere3->setMaterial(sphereMaterial3);
   
   scene.addSurface(sphere3);

   Sphere* sphere4 = new Sphere(glm::vec3(0.0f, 0.0f, 5.0f), 1.5f);
   sphere4->setTransformation(M);
   sphere4->setMaterial(sphereMaterial4);

   //scene.addSurface(sphere4);



   char* triangleMeshFileName = "objmodels/monkey.obj";
   TriangleMesh* mesh = new TriangleMesh();
   glm::mat4 meshTransformation = glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 0.0f, -1.0f));
   
   mesh->setTransformation(meshTransformation);
   mesh->setMaterial(triangleMeshMaterial);
   
   mesh->loadFromFile(triangleMeshFileName);
   scene.addTriangleMesh(mesh);
   

   std::cout << "Triangle Mesh (" << triangleMeshFileName << ") has " << mesh->getNumTriangles() << " Triangles." << std::endl;


   char* triangleMeshGridFileName = "objmodels/grid.obj";
   TriangleMesh* meshGrid = new TriangleMesh();
   glm::mat4 meshGridTransformation = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.5f, 0.0f));
   
   meshGrid->setTransformation(meshGridTransformation);
   meshGrid->setMaterial(gridMaterial);
   
   meshGrid->loadFromFile(triangleMeshGridFileName);
   scene.addTriangleMesh(meshGrid);


     char* triangleMeshGrid1FileName = "objmodels/grid_back.obj";
   TriangleMesh* meshGrid1 = new TriangleMesh();
   glm::mat4 meshGrid1Transformation = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 5.7f, 0.0f));
   
   meshGrid1->setTransformation(meshGrid1Transformation);
   meshGrid1->setMaterial(gridMaterialLeft);
   
   meshGrid1->loadFromFile(triangleMeshGrid1FileName);
   scene.addTriangleMesh(meshGrid1);


   /*
   float diskRadiusSquared = 2.0f;
   glm::vec3 diskPlanePoint(0.0f, -1.0f, 0.0f);
   glm::vec3 diskNormal(0.0f, 1.0f, 0.0f);

   Disk *disk = new Disk(diskRadiusSquared, diskPlanePoint, diskNormal);
   disk->setTransformation(glm::mat4(1.0f));

   gridMaterial.setReflective(true);
   disk->setMaterial(gridMaterial);

 */
   //scene.addSurface(disk);


   scene.addLightSource(lightSource);
   scene.addLightSource(lightSource1);
   //scene.addLightSource(lightSource2);

   // flush changes
   double bvh_start,bvh_end,bvh_diff;
   double bvh_msecs;
   int bvh_hours,bvh_minutes,bvh_seconds;

   std::cout << "Constructring Scene.Building BVH..." << std::endl;
   bvh_start = getRealTime();
   scene.flush();
   
   bvh_end = getRealTime();

   bvh_diff = bvh_end - bvh_start;

   bvh_minutes = bvh_diff / 60;
   bvh_seconds = ((int)bvh_diff) % 60;
   bvh_msecs = bvh_diff * 1000;

   std::cout << "BVH construction completed.Scene Ready in " << bvh_minutes << " minutes and " << bvh_seconds << " seconds" << std::endl;
   
   if(renderOnce){
      std::cout << "Rendering Once to Image file..." << std::endl;
      double start,end,diff;
      double msecs;
      int hours,minutes,seconds;
      start = getRealTime();
      scene.render(camera, imageBuffer);
      end = getRealTime();

      diff = end - start;

      minutes = diff / 60;
      seconds = ((int)diff) % 60;
      msecs = diff * 1000;

      //write_ppm(WINDOW_WIDTH, WINDOW_HEIGHT, imageBuffer);

      // make image (0,0) top left corner
      // Start at the end of the image buffer and use negative stride
      stbi_write_png("rayTracedImageAA.png", WINDOW_WIDTH, WINDOW_HEIGHT, 4, imageBuffer + WINDOW_WIDTH * WINDOW_HEIGHT * 4, -WINDOW_WIDTH*4);
      std::cout << "Rendering Once: Completed in " << minutes << "minutes, " << seconds << "sec " << std::endl;
      std::cout << "That's About " << msecs << "ms" << std::endl;
   }
   else{
   while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window))
   {
      currTime = glfwGetTime();
      memset(imageBuffer, 0, sizeof(unsigned char) * 4 * WINDOW_WIDTH * WINDOW_HEIGHT);
      // handle input
      double xpos, ypos;
      glfwGetCursorPos(window, &xpos, &ypos);
      glfwSetCursorPos(window, WINDOW_WIDTH/2.0, WINDOW_HEIGHT/2.0);

      horizontalAngle += mouseSpeed * float(WINDOW_WIDTH/2 - xpos );
      verticalAngle   -= mouseSpeed * float( WINDOW_HEIGHT/2 - ypos );

      if( verticalAngle > maxAbsoluteVerticalAngle)
         verticalAngle = maxAbsoluteVerticalAngle;
      else if( verticalAngle < -maxAbsoluteVerticalAngle)
         verticalAngle = -maxAbsoluteVerticalAngle;

      //glm::vec3 viewDirection(cos(verticalAngle) * sin(horizontalAngle),
      //                    sin(verticalAngle),
      //                    cos(verticalAngle) * cos(horizontalAngle));

      glm::mat4 viewTransformationX = glm::rotate(glm::mat4(1.0f), horizontalAngle, glm::vec3(0.0f, 1.0f, 0.0f));
      glm::mat4 viewTransformation = glm::rotate(viewTransformationX, verticalAngle, glm::vec3(1.0f, 0.0f, 0.0f));

      glm::vec3 viewDirection  = glm::vec3( viewTransformation * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));
      glm::vec3 rightDirection = glm::normalize( glm::cross(viewDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
      glm::vec3 upDirection = glm::normalize(glm::cross(rightDirection, viewDirection)); 

      // Right vector
      //glm::vec3 rightDirection = glm::normalize(glm::vec3(sin(horizontalAngle - 3.1415f/2.0f),0, cos(horizontalAngle - 3.1415f/2.0f)));
      //glm::vec3 upDirection    = glm::normalize(glm::cross( viewDirection, rightDirection));

      //rightDirection   = glm::normalize(glm::cross(viewDirection, upDirection));

      glm::vec3 cameraPos = camera.getPosition();
      // Move forward
      if (glfwGetKey(window, GLFW_KEY_W ) == GLFW_PRESS){
         cameraPos += viewDirection * speed * deltaTime;
      }
      // Move backward
      if (glfwGetKey(window, GLFW_KEY_S ) == GLFW_PRESS){
         cameraPos -= viewDirection * speed * deltaTime;   
      }
      // Strafe right
      if (glfwGetKey(window, GLFW_KEY_D ) == GLFW_PRESS){
         cameraPos += rightDirection * speed * deltaTime;
      }
      // Strafe left
      if (glfwGetKey(window, GLFW_KEY_A ) == GLFW_PRESS){
         cameraPos -= rightDirection * speed * deltaTime;
      }



      // update camera
      camera.setViewingDirection(viewDirection);
      camera.setUpVector(upDirection);
      camera.setRightVector(rightDirection);
      camera.setPosition(cameraPos);


      /* Raytracing */
      scene.render(camera, imageBuffer);
      

      // copy pixels to GPU buffer (async copy)
      glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, sizeof(GLchar) * WINDOW_WIDTH * WINDOW_HEIGHT * 4, imageBuffer);
      
      // update texture data from the GL_PIXEL_UNPACK_BUFFER bounded pixel buffer object
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0 , 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      

      /* draw window sized quad */
      glDrawArrays( GL_TRIANGLES, 0, 6);

      glfwSwapBuffers(window);
      fps++;
      // measure Frames per second
      deltaTime = glfwGetTime() - currTime;
      timeAcc += glfwGetTime() - currTime;
      if( timeAcc >= 1.0){
         timeAcc = 0.0;
         std::cout << " FPS: " << fps << std::endl;
         fps = 0;
      }

      glfwPollEvents();

   }


   glDeleteProgram(program);
   }
   glfwTerminate();
   return 0;
}

