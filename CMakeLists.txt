cmake_minimum_required(VERSION 3.15) #设置cmake最低要求
project(
    main      #项目名称
    LANGUAGES CXX C     #编程语言
    DESCRIPTION "模型推理集合"
)
set(CMAKE_CXX_STANDARD 23) # 设置为C++ 23标准
# set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求使用C++23

get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin)
set(CMAKE_BUILD_PARALLEL_LEVEL 4) 

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/algorithm
    ${CMAKE_SOURCE_DIR}/include/mtool
    ${CMAKE_SOURCE_DIR}/include/clipper2
    ${CMAKE_SOURCE_DIR}/include/bytetrack/include/
)
# 设定项目编写的所有代码文件路径
file(GLOB SRC_FILES 
    CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/*/*.h"
    "${PROJECT_SOURCE_DIR}/include/*/*.hpp"
    "${PROJECT_SOURCE_DIR}/include/*/*.cpp"
    "${PROJECT_SOURCE_DIR}/include/*/*.hxx"
    "${PROJECT_SOURCE_DIR}/include/bytetrack/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/bytetrack/src/*.cpp"
)

find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME} TBB::tbb)
target_link_libraries(${CMAKE_PROJECT_NAME} onnxruntime)
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBRARIES})
