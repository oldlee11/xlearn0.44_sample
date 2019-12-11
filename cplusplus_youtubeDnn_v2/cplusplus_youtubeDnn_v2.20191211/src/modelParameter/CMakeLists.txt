set(SUBDIRNAME modelParameter)
set(TESTFILE null)


# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/${SUBDIRNAME})


# Build static library
set(STA_DEPS base reader)
add_library(${SUBDIRNAME} STATIC parameters.cpp)
if(NOT WIN32)
    target_link_libraries(${SUBDIRNAME} ${STA_DEPS})
else(WIN32)
    target_link_libraries(${SUBDIRNAME} ${STA_DEPS} Ws2_32)
endif()


# Build uinttests.
if(NOT ${TESTFILE} MATCHES "null")
    if(NOT WIN32)
        set(LIBS ${STA_DEPS} ${SUBDIRNAME} pthread)
    else(WIN32)
        set(LIBS ${STA_DEPS} ${SUBDIRNAME})
    endif()
    add_executable(${TESTFILE} ${TESTFILE}.cpp)
    target_link_libraries(${TESTFILE} ${LIBS})
endif()


# Install library and header files
install(TARGETS ${SUBDIRNAME} DESTINATION lib/${SUBDIRNAME})
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/${SUBDIRNAME})