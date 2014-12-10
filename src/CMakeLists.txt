find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()



find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)
find_package(ASSIMP REQUIRED)
include_directories(${GLEW_INLUDE_PATH})
include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(glm/glm)
include_directories(${Eos_SOURCE_DIR}/glfw-3.0.4/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_20 -maxrregcount 40)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -arch=sm_20)
#SET(CUDA_NVCC_FLAGS "-arch=sm_30 -std=c++11" CACHE STRING "nvcc flags" FORCE)

#SET(CUDA_SEPARABLE_COMPILATION ON)
#LIST(APPEND CUDA_NVCC_FLAGS -arch sm_20)
#LIST(APPEND CUDA_NVCC_FLAGS -arch sm_30)
#LIST(APPEND CUDA_NVCC_FLAGS -std c++11)

#LIST(APPEND CMAKE_CXX_FLAGS "-O3 -ffast-math -Wall")


#LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
SET(CUDA_SEPARABLE_COMPILATION ON)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

#LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
#LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_35)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11)


set(BUILD_SHARED_LIBS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
FILE(GLOB CPPSources *.cpp)
FILE(GLOB CUSources *.cu)

CUDA_ADD_EXECUTABLE(Eos ${CPPSources} ${CUSources})


#cuda_add_library(cudautilities ${CUSources} STATIC)


#cuda_add_executable(Eos ${CPPSources})
target_link_libraries(Eos glfw ${GLFW_LIBRARIES} ${GLEW_LIBRARY} ${ASSIMP_LIBRARIES})
