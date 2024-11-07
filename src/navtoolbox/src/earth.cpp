#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "earth.hpp"

namespace py = pybind11;

PYBIND11_MODULE(earth, m) {
    m.doc() = "Earth geometry and gravity models using WGS84 parameters";

    m.attr("RATE") = earth::RATE;
    m.attr("A") = earth::A;
    m.attr("E2") = earth::E2;
    m.attr("GE") = earth::GE;
    m.attr("GP") = earth::GP;
    m.attr("F") = earth::F;
    m.def("say_hello", &earth::say_hello, "A function that prints says hello.");
    m.def("principal_radii", &earth::principal_radii, py::arg("lat"), py::arg("alt"),
          "Computes the principal radii of curvature of Earth ellipsoid.");
    m.def("gravity", &earth::gravity, py::arg("lat"), py::arg("alt"),
          "Computes the gravity magnitude using the Somigliana model with linear altitude correction.");
    m.def("gravity_n", &earth::gravity_n, py::arg("lat"), py::arg("alt"),
          "Computes the gravity vector in the NED (North-East-Down) frame.");
    //m.def("gravitation_ecef", &earth::gravitation_ecef, py::arg("lla"),
    //      "Computes the gravitational force vector in ECEF frame.");
    m.def("curvature_matrix", &earth::curvature_matrix, py::arg("lat"), py::arg("alt"),
          "Computes the Earth curvature matrix.");
    m.def("rate_n", &earth::rate_n, py::arg("lat"),
          "Computes Earth rate in the NED frame.");
}