# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/base)

# Build static library
add_library(base STATIC split_string.cpp timer.cpp)

# Build unittests.
if(NOT WIN32)
    set(LIBS base pthread)
else(WIN32)
    set(LIBS base)
endif()



# Install library and header files
install(TARGETS base DESTINATION lib/base)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/base)