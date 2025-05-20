cmake_minimum_required(VERSION 3.20)
project(yolo_ncnn)

# set(OpenCV_DIR ~/opencv-mobile-4.10.0-ubuntu-2204/lib/cmake/opencv4)
# find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs highgui objdetect)

if(OpenCV_FOUND)
    message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

    if(${OpenCV_VERSION_MAJOR} GREATER 3)
        set(CMAKE_CXX_STANDARD 11)
    endif()
endif()

link_directories("/opt/homebrew/lib")

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE "/opt/homebrew/include")

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ncnn yaml-cpp SPIRV ${OpenCV_LIBS})

# OGLCompiler

# Copy model files and test image to build directory
set(MODEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/yolo12n_opt.param
    ${CMAKE_CURRENT_SOURCE_DIR}/yolo12n_opt.bin
)

foreach(FILE ${MODEL_FILES})
    if(EXISTS ${FILE})
        file(COPY ${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    else()
        message(WARNING "File not found: ${FILE}")
    endif()
endforeach()