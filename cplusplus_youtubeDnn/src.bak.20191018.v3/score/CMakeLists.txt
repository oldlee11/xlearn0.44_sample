# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/score)

# Build static library
set(STA_DEPS data base)
add_library(score STATIC score_function.cc youtubeDnn_score.cc)
target_link_libraries(score ${STA_DEPS})

# Build uinttests
set(LIBS score data base gtest)

# Install library and header files
install(TARGETS score DESTINATION lib/score)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/score)
