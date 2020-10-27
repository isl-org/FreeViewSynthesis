#pragma once

#include <functional>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

typedef py::array_t<float, py::array::c_style | py::array::forcecast>
    array_f32_d;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> array_i32_d;
typedef py::array_t<long, py::array::c_style | py::array::forcecast>
    array_i64_d;

typedef Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>
    depthmap_t;
typedef Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> mat3_t;
typedef Eigen::Map<const Eigen::Matrix<float, 3, 1>> vec3_t;
typedef Eigen::Map<const Eigen::Matrix<float, -1, 3, Eigen::RowMajor>> points_t;
typedef Eigen::Matrix<float, 3, 4> proj_t;

template <typename T>
py::array_t<T> create_array1(const std::vector<T>& data) {
    T* new_data = new T[data.size()];
    std::memcpy(new_data, data.data(), data.size() * sizeof(T));
    py::capsule free_data(new_data, [](void* f) {
        float* new_data = reinterpret_cast<float*>(f);
        delete[] new_data;
    });
    return py::array_t<T>({data.size()}, new_data, free_data);
}

template <typename T>
py::array_t<T> create_array2(const std::vector<T>& data, int height,
                             int width) {
    if (int(data.size()) != height * width) {
        throw std::invalid_argument("invalid size in create_array2");
    }
    T* new_data = new T[data.size()];
    std::memcpy(new_data, data.data(), data.size() * sizeof(T));
    py::capsule free_data(new_data, [](void* f) {
        float* new_data = reinterpret_cast<float*>(f);
        delete[] new_data;
    });
    return py::array_t<T>({height, width}, new_data, free_data);
}

template <typename T>
py::array_t<T> create_arrayN(const std::vector<T>& data, const std::vector<int>& shape) {
    if (int(data.size()) != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<float>())) {
        throw std::invalid_argument("invalid size in create_arrayN");
    }
    T* new_data = new T[data.size()];
    std::memcpy(new_data, data.data(), data.size() * sizeof(T));
    py::capsule free_data(new_data, [](void* f) {
        float* new_data = reinterpret_cast<float*>(f);
        delete[] new_data;
    });
    return py::array_t<T>(shape, new_data, free_data);
}

