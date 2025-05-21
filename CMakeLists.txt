cmake_minimum_required(VERSION 3.20)
project(VulkanSamples)

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)

set(SELECTED_PLATFORM "windows")

file(GLOB_RECURSE SOURCES "main.cpp" "include/*.hpp" "src/*.cpp")

add_executable(VulkanSamples ${SOURCES})
target_include_directories(VulkanSamples PUBLIC ${Vulkan_INCLUDE_DIRS} "include")

# Compiling the shaders
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/vert.spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/frag.spv
    COMMAND glslc -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/vert.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/simple.vert
    COMMAND glslc -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/frag.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/simple.frag
    DEPENDS shaders/simple.vert shaders/simple.frag
)

add_custom_target(shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shaders/vert.spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/frag.spv)