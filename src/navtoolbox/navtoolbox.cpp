#include <iostream>
#include <pybind11/pybind11.h>

void say_hello() {
    std::cout << "Hello World! This is C++!" << std::endl;
}

PYBIND11_MODULE(navtoolbox, m) {
    m.def("say_hello", &say_hello, "A function that prints says hello.");
};