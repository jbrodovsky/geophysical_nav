#include <iostream>
#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "earth.hpp"
#include "transform.hpp"
#include "util.hpp"
#include "integrate.hpp"

namespace navtoolbox{
    void say_hello() {
        std::cout << "Hello World! This is C++!" << std::endl;
    }
}

namespace py = pybind11;

PYBIND11_MODULE(navtoolbox, m) {
      // Top level navtoolbox module
      m.doc() = "Navigation toolbox for coordinate transformations and Earth models";
      m.def("say_hello", &navtoolbox::say_hello, "A function that prints says hello.");
      // Earth submodule from earth.hpp
      py::module earth_module = m.def_submodule("earth", "Earth geometry and gravity models using WGS84 parameters");
      earth_module.attr("RATE") = earth::RATE;
      earth_module.attr("A") = earth::A;
      earth_module.attr("E2") = earth::E2;
      earth_module.attr("GE") = earth::GE;
      earth_module.attr("GP") = earth::GP;
      earth_module.attr("F") = earth::F;
      earth_module.def("say_hello", &earth::say_hello, "A function that prints says hello.");
      earth_module.def("principal_radii", &earth::principal_radii, py::arg("lat"), py::arg("alt"),
            "Computes the principal radii of curvature of Earth ellipsoid.");
      earth_module.def("gravity", &earth::gravity, py::arg("lat"), py::arg("alt"),
            "Computes the gravity magnitude using the Somigliana model with linear altitude correction.");
      earth_module.def("gravity_n", &earth::gravity_n, py::arg("lat"), py::arg("alt"),
            "Computes the gravity vector in the NED (North-East-Down) frame.");
      earth_module.def("gravitation_ecef", &earth::gravitation_ecef, py::arg("lla"),
            "Computes the gravitational force vector in ECEF frame.");
      earth_module.def("curvature_matrix", &earth::curvature_matrix, py::arg("lat"), py::arg("alt"),
            "Computes the Earth curvature matrix.");
      earth_module.def("rate_n", &earth::rate_n, py::arg("lat"),
            "Computes Earth rate in the NED frame.");
      // Transform submodule from transform.hpp
      py::module transform_module = m.def_submodule("transform", "Coordinate transformations for navigation toolbox");
      transform_module.doc() = "Coordinate transformations for navigation toolbox";
      transform_module.attr("DEG_TO_RAD") = transform::DEG_TO_RAD;
      transform_module.attr("RAD_TO_DEG") = transform::RAD_TO_DEG;
      //transform_module.def("lla_to_ecef", &transform::lla_to_ecef, py::arg("lla"),
      //      "Converts latitude, longitude, and altitude to ECEF Cartesian coordinates.");
      transform_module.def("ecef_to_lla", py::overload_cast<const Eigen::Vector3d&>(&transform::ecef_to_lla), 
            "Convert a single ECEF coordinate to latitude, longitude, altitude.");
      transform_module.def("ecef_to_lla", py::overload_cast<const std::vector<Eigen::Vector3d>&>(&transform::ecef_to_lla), 
            "Convert a list of ECEF coordinates to latitude, longitude, altitude.");
      transform_module.def("lla_to_ned", &transform::lla_to_ned, py::arg("lla"), py::arg("ref_lla"),
            "Converts latitude, longitude, and altitude to NED Cartesian coordinates.");
      transform_module.def("ned_to_lla", &transform::ned_to_lla, py::arg("ned"), py::arg("ref_lla"),
            "Converts NED Cartesian coordinates to latitude, longitude, and altitude.");
      transform_module.def("mat_en_from_ll", &transform::mat_en_from_ll, py::arg("lat"), py::arg("lon"),
            "Computes the transformation matrix from ECEF to NED frame.");
      transform_module.def("mat_from_rph", py::overload_cast<double, double, double>(&transform::mat_from_rph), py::arg("roll"), py::arg("pitch"), py::arg("heading"),
            "Computes the transformation matrix from body to NED frame.");
      transform_module.def("mat_from_rph", &transform::mat_to_rph, py::arg("rph"),
            "Converts a orientation vector of roll, pitch, and heading angles to a rotation matrix.");
      transform_module.def("mat_from_rotvec", &transform::mat_from_rotvec, py::arg("rv"),
                  "Converts a rotation vector to a rotation matrix.");
      transform_module.def("rotvec_from_mat", &transform::rotvec_from_mat, py::arg("mat"),
                  "Converts a rotation matrix to a rotation vector.");  
      transform_module.def("say_hello", &transform::say_hello, "A function that prints says hello.");
      // Util submodule from util.hpp
      py::module util_module = m.def_submodule("util", "Utility functions for navigation toolbox");
      util_module.doc() = "Utility functions for navigation toolbox";
      util_module.def("mm_prod", &util::mm_prod, py::arg("a"), py::arg("b"), py::arg("at") = false, py::arg("bt") = false);
      util_module.def("mm_prod_symmetric", &util::mm_prod_symmetric, py::arg("a"), py::arg("b"));
      util_module.def("mv_prod", &util::mv_prod, py::arg("a"), py::arg("b"), py::arg("at") = false);
      util_module.def("skew_matrix", &util::skew_matrix, py::arg("vec"));
      util_module.def("compute_rms", &util::compute_rms, py::arg("data"));
      util_module.def("to_180_range", (double (*)(double)) &util::to_180_range, py::arg("angle"));
      // Integrate submodule from integrate.hpp
      py::module integrate_module = m.def_submodule("integrate", "Strapdown inertial navigation integration");
      integrate_module.doc() = "Strapdown inertial navigation integration";
      py::class_<StrapdownIntegrator>(integrate_module, "StrapdownIntegrator")
            .def(py::init<double, double, double, double, double, double, double, double, double>())
            .def("integrate", &StrapdownIntegrator::integrate, py::arg("dt"), py::arg("gyro"), py::arg("accel"));
};