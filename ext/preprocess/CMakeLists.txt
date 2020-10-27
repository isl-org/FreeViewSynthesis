cmake_minimum_required(VERSION 3.1)
project(preprocess)

set(CMAKE_CXX_STANDARD 14)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_subdirectory(pybind11)

pybind11_add_module(preprocess 
    main.cpp 
    preprocess.cpp 
)
