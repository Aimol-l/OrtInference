cmake_minimum_required(VERSION 3.15) #设置cmake最低要求
project(
    main      #项目名称
    LANGUAGES CXX C     #编程语言
    DESCRIPTION "table detect demo"
)
set(CMAKE_CXX_STANDARD 23) # 设置为C++ 23标准
set(CMAKE_BUILD_TYPE "Release" )
# set(CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求使用C++23


# 获取 CMakeLists.txt 文件所在目录的绝对路径
get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin)
# 四个线程编译
set(CMAKE_BUILD_PARALLEL_LEVEL 8) 


include_directories(${PROJECT_DIR}/include/)
# 设定项目编写的所有代码文件路径
file(GLOB SRC_FILES 
    CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})


target_link_libraries(${CMAKE_PROJECT_NAME} TBB::tbb)
target_link_libraries(${CMAKE_PROJECT_NAME} onnxruntime)
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBRARIES})
