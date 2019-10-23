# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/reader)

# Build static library
set(STA_DEPS data base)
add_library(reader STATIC parser.cc file_splitor.cc reader.cc)
target_link_libraries(reader ${STA_DEPS})

# Build uinttests.
if(NOT WIN32)
set(LIBS reader data base pthread gtest)
else(WIN32)
set(LIBS reader data base gtest)
endif()



# Install library and header files
install(TARGETS reader DESTINATION lib/reader)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/reader)
