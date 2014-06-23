#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "stb_image.h"
#include "Ray.h"
#include "sphere.h"
#include "RayIntersection.h"
#include "triangleMesh.h"
#include "Camera.h"
#include "Scene.h"
#include "LightSource.h"
#include "Material.h"
#include "Plane.h"

#define WINDOW_WIDTH   640  // in pixels
#define WINDOW_HEIGHT  480  // in pixels
#define FOV            70

#ifndef EPSILON
#define EPSILON        1e-3
#endif

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
 
   // Check the program
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
   GLFWwindow* window;
   glfwSetErrorCallback(glfwErrorCallback);

   if(!glfwInit())
   {
      std::cerr << "Failed to initialize GLFW...\n";
      return -1;
   }

   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

   window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "RayTracing", NULL, NULL);
   if(!window)
   {
      std::cerr << "Failed to open GLFW window...\n";
      glfwTerminate();
      return -1;
   }

   std::cout<< "Going to Create context\n";
   glfwMakeContextCurrent(window);

   std::cout << "Going to initialize GLEW...\n";
   glewExperimental = GL_TRUE;
   if (glewInit() != GLEW_OK)
   {
      std::cerr << "Failed to initialize GLEW...\n";
      glfwTerminate();
      return -1;
   }
  
   std::cout << "Going to Create Vertex Array Object...\n";
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   // set up shaders
   GLuint program;

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

   GLuint vbo;
   glGenBuffers(1, &vbo);

   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);   // copy to device memory

   unsigned char* imageBuffer = new unsigned char[ WINDOW_WIDTH * WINDOW_HEIGHT * 4];
   memset(imageBuffer, 0, sizeof(unsigned char) * 4 * WINDOW_WIDTH * WINDOW_HEIGHT);

   GLuint pbo;
   glGenBuffers(1, &pbo);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
   glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(GLchar) * WINDOW_WIDTH * WINDOW_HEIGHT * 4, imageBuffer, GL_DYNAMIC_DRAW);

   /*
   int texx, texy, n;
   int force_channels = 4;
   unsigned char* image_data = stbi_load (argv[1], &texx, &texy, &n, force_channels);
   if( !image_data ){
      std::cout<< "Image no found\n";
      glfwTerminate(); 
   }

   int width_in_bytes = texx * 4;
   unsigned char *top = NULL;
   unsigned char *bottom = NULL;
   unsigned char temp = 0;
   int half_height = texy / 2;

   for (int row = 0; row < half_height; row++) {
      top = image_data + row * width_in_bytes;
      bottom = image_data + (texy - row - 1) * width_in_bytes;
      for (int col = 0; col < width_in_bytes; col++) {
         temp = *top;
         *top = *bottom;
         *bottom = temp;
         top++;
         bottom++;
      }
   }

 */

   // generate texture
   GLuint texture;
   glGenTextures(1, &texture);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);


   glUseProgram(program);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);



   glm::vec4 sphereOrigin(0.0, 0.0, 0.0, 1.0);
   glm::vec4 sphereColor( 0.6f , 0.6f , 0.6f, 0.0);

   glm::vec4 lightPosition( 120.0f, 0.0f , 0.0f, 1.0f);
   glm::vec4 lightColor( 1.0, 1.0, 1.0, 0.0);
   
   float fovx_rad = FOV * M_PI / (float) 180.0;
   float fovy_rad =  FOV * M_PI / (float)180.0;

   float tan_fovx = tan( fovx_rad / (float) 2);
   float tan_fovy = tan( fovy_rad / (float) 2);
   std::cout << "FOVX = " << fovx_rad << " FOVY = " << fovy_rad << std::endl;
   std::cout << "tan_FOVX = " << tan_fovx << " tan_FOVY = " << tan_fovy << std::endl;


   glm::mat4 M = glm::translate( glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
   glm::mat4 identity(1.0f);
   glm::mat4 mat_rot = glm::rotate(identity, 0.1f, glm::vec3(0.0f, 0.0f,1.0f));

  

   // transfer pixels to texture via pixel buffer object
   glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH , WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
   // filtering
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

   Camera camera;

   camera.setPosition(glm::vec3(0.0f, 0.0f,0.0f));
   camera.setViewingDirection(glm::vec3(0.0f,0.0f, -1.0f));
   camera.setRightVector(glm::vec3(1.0f, 0.0f, 0.0f));
   camera.setUpVector(glm::vec3(0.0f, 1.0f, 0.0f));
   camera.setFov(FOV);
   camera.setWidth(WINDOW_WIDTH);
   camera.setHeight(WINDOW_HEIGHT);

   float horizontalAngle = 0.0f;
   float verticalAngle = 45.0f;
   const float maxAbsoluteVerticalAngle = 3.1415f / 2.0f - 0.001f;
   float speed = 5.0f; // 3 units / second
   float mouseSpeed = 0.005f;

   glfwSetCursorPos(window, WINDOW_WIDTH/2.0, WINDOW_HEIGHT/2.0);


   double currTime = 0.0;
   double timeAcc = 0.0;
   int fps = 0;
   float deltaTime = 0.0f;

   // set up scene
   Scene scene;
   Material sphereMaterial(0.1f, glm::vec4(0.5f, 0.5f, 0.5f, 0.0f), glm::vec4(1.0f), 40);
   Material sphereMaterial1(0.1f, glm::vec4(1.f, 0.0f, 0.0f, 0.0f), glm::vec4(1.0f), 40);
   Material gridMaterial(0.1f, glm::vec4(0.8f, 0.8f, 0.7f, 0.0f), glm::vec4(0.0f), 40);
   LightSource* lightSource  = new LightSource(glm::vec4( 100.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f));
   LightSource* lightSource1 = new LightSource(glm::vec4( 0.0f, -100.0f, 0.0f, 1.0f), glm::vec4(1.0f));

   Sphere* sphere = new Sphere(glm::vec3(0.0f), 2);
   sphere->setTransformation(M);
   sphere->setMaterial(sphereMaterial);

   scene.addSurface(sphere);

   Sphere* sphere1 = new Sphere(glm::vec3(4.0f, 0.0f, 0.0f), 2);
   sphere1->setTransformation(M);
   sphere1->setMaterial(sphereMaterial1);

   scene.addSurface(sphere1);
   
   /*
   TriangleMesh* grid = new TriangleMesh();
   glm::mat4 gridTransformation = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -5.0f, 0.0f));
   grid->loadFromFile("triangle_grid.obj");
   grid->setTransformation(gridTransformation);
   grid->setMaterial(gridMaterial);

   */
   //scene.addSurface(grid);
 
   scene.addLightSource(lightSource);
   scene.addLightSource(lightSource1);

   while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window))
   {
      currTime = glfwGetTime();
      lightPosition = mat_rot * lightPosition;
      memset(imageBuffer, 0, sizeof(unsigned char) * 4 * WINDOW_WIDTH * WINDOW_HEIGHT);
      // handle input
      double xpos, ypos;
      glfwGetCursorPos(window, &xpos, &ypos);
      glfwSetCursorPos(window, WINDOW_WIDTH/2.0, WINDOW_HEIGHT/2.0);

      horizontalAngle += mouseSpeed * float(WINDOW_WIDTH/2 - xpos );
      verticalAngle   += mouseSpeed * float( WINDOW_HEIGHT/2 - ypos );

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
   glfwTerminate();
   return 0;
}

