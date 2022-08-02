set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fext-numeric-literals ${CMAKE_CXX_FLAGS}")

# libjsoncpp.so
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)
add_library(jsoncpp SHARED ${SRCS})
target_include_directories(jsoncpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(jsoncpp  PRIVATE -std=c++11 -Wall -O3 -fPIC)

# Install libWKTParser.so
install(TARGETS jsoncpp DESTINATION lib)