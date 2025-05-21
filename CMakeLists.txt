cmake_minimum_required(VERSION 3.30)

project(VulkanSamples LANGUAGES CXX)

find_package(Vulkan REQUIRED)

set(SELECTED_PLATFORM "windows")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(VulkanSamples)

target_sources(VulkanSamples
    PRIVATE
        main.cpp)

target_sources(VulkanSamples
    PRIVATE
        FILE_SET CXX_MODULES
        FILES
            Application.cppm)

target_include_directories(VulkanSamples PUBLIC ${Vulkan_INCLUDE_DIRS} "include" ${modules})

# Compiling the shaders
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/vert.spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/frag.spv
    COMMAND glslc -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/vert.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/simple.vert
    COMMAND glslc -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/frag.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/simple.frag
    DEPENDS shaders/simple.vert shaders/simple.frag
)

add_custom_target(shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shaders/vert.spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/frag.spv)