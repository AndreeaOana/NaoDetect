cmake_minimum_required(VERSION 3.0)

# define the project name
set(project_name "PRJ3-008284-010097")

# set the project name
project("${project_name}")


# add opencv package to the project
#SET(OPENCV_ROOT_DIR "/home/aop/Desktop/OpenCV/opencv-3.2.0")
#SET(OPENCV_INC_DIR "${OPENCV_ROOT_DIR}/include/opencv")
#SET(OPENCV_LIB_DIR "${OPENCV_ROOT_DIR}/build/lib")
find_package(OpenCV 3 REQUIRED)
# additional packages required
find_package(Boost COMPONENTS system filesystem REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#add OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS} include/)

add_executable("${project_name}"
        PRJ34-S008284-S010097.cpp
        include/ImageReader.h src/ImageReader.cpp include/ImageReader.h
        src/ImageReader.cpp include/ImageReader.h 
        src/BagOfSIFT.cpp include/BagOfSIFT.h src/SVMAnalysis.cpp include/SVMAnalysis.h src/Evaluator.cpp include/Evaluator.h)

target_link_libraries( "${project_name}" ${OpenCV_LIBS})
target_link_libraries("${project_name}"
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY})
