cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
project (swDNN)
set (swim2coltest_VERSION_MAJOR 1)
set (swim2coltest_VERSION_MINOR 0)


set(CMAKE_C_COMPILER "sw5cc.new")
set(CMAKE_CXX_COMPILER "mpiCC")
set(CMAKE_AR "swar")
option(BUILD_TEST "Build the test sources" ON)

file(GLOB SLAVE_SOURCES "./slave/tensor_trans/*.c" "./slave/im2col/*.c"
  "./slave/col2im/*.c" "./slave/bias/*.c" "./slave/winograd/*.c"
  "./slave/implicit_conv/*.c" "./slave/batch_norm/*.c"
  "./slave/pooling/*.c" "./slave/softmax/*.c" "./slave/scale/*.c" "./slave/lstm/*.c")
file(GLOB HOST_SOURCES "./src/*.c")
file(GLOB TEST_SOURCES "./unitest/src/*.c" "./unitest/src/*.cpp")

add_definitions(-msimd -O2)
add_definitions(-D_MEM_ALIGN_)

set_source_files_properties(${SLAVE_SOURCES} COMPILE_FLAGS "-slave")
set_source_files_properties(${HOST_SOURCES} ${TEST_SOURCES} COMPILE_FLAGS "-host")
set_source_files_properties($