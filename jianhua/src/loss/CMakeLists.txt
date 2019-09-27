# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/loss)

# Build static library
set(STA_DEPS score data base)
add_library(loss STATIC loss.cc squared_loss.cc 
cross_entropy_loss.cc metric.cc)
target_link_libraries(loss ${STA_DEPS})

# Build uinttests
set(LIBS loss score data base gtest)



# Install library and header files
install(TARGETS loss DESTINATION lib/loss)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/loss)
