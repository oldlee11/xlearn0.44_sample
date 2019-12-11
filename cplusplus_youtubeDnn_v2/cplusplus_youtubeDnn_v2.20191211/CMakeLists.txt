project ("youtubeDnn")
cmake_minimum_required(VERSION 2.8)

#-------------------------------------------------------------------------------
# Take almost all warnings;
# Do not generate debug symbols;
# Optimization level 3;
# Using c++11;
# Optimization on architecture
#-------------------------------------------------------------------------------
if(NOT WIN32)
    add_definitions("-Wall -Wno-sign-compare -O3 -std=c++11 -march=native -Wno-strict-aliasing -Wno-comment")
    #add_definitions("-Wall -Wno-sign-compare -O0 -std=c++11 -march=native -Wno-strict-aliasing -Wno-comment")
else(WIN32)
    add_definitions("/WX- /MT")
endif()

#-------------------------------------------------------------------------------
# Declare where our project will be installed.
#-------------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "youtubeDnn")

#-------------------------------------------------------------------------------
# For Mac OSX
#-------------------------------------------------------------------------------
#set(CMAKE_MACOSX_RPATH TRUE)

#-------------------------------------------------------------------------------
# Set include paths and library paths.
#-------------------------------------------------------------------------------
include_directories(
        "${PROJECT_SOURCE_DIR}"
        "${PROJECT_BINARY_DIR}"
)

link_directories(
        "${PROJECT_BINARY_DIR}/src/base"
        "${PROJECT_BINARY_DIR}/src/reader"
        "${PROJECT_BINARY_DIR}/src/modelParameter"
        "${PROJECT_BINARY_DIR}/src/network"
        "${PROJECT_BINARY_DIR}/src/loss"
        "${PROJECT_BINARY_DIR}/src/solver"
)

#-------------------------------------------------------------------------------
# Declare packages in xLearn project.
#-------------------------------------------------------------------------------
add_subdirectory(src/base)
add_subdirectory(src/reader)
add_subdirectory(src/modelParameter)
add_subdirectory(src/network)
add_subdirectory(src/loss)
add_subdirectory(src/solver)





