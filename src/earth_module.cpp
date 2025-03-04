#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "include/earth.hpp"
namespace py = pybind11;

PYBIND11_MODULE(earth, m) {
    m.doc() = "WGS84 Earth model class definition and coordinate transformations";
    m.attr("RATE") = py::cast(RATE);
    m.attr("EQUATORIAL_RADIUS") = py::cast(EQUATORIAL_RADIUS);
    m.attr("POLAR_RADIUS") = py::cast(POLAR_RADIUS);
    m.attr("ECCENTRICITY") = py::cast(ECCENTRICITY);
    m.attr("ECCENTRICITY_SQUARED") = py::cast(ECCENTRICITY_SQUARED);
    m.attr("GE") = py::cast(GE);
    m.attr("GP") = py::cast(GP);
    m.attr("f") = py::cast(f);
    m.attr("F") = py::cast(F);
    m.def("eci_to_ecef", &ECIToECEF, "Rotation matrix from ECI to ECEF");
    m.def("ecef_to_ned", &ECEFToNED, "Rotation matrix from ECEF to NED");
    m.def("ned_to_body", &NEDToBody, "Rotation matrix from NED to body");
    m.def("ecef_to_eci", &ECEFToECI, "Rotation matrix from ECEF to ECI");
    m.def("ned_to_ecef", &NEDToECEF, "Rotation matrix from NED to ECEF");
    m.def("body_to_ned", &BodyToNED, "Rotation matrix from body to NED");
    m.def("vector_to_skew_symmetric", &vectorToSkewSymmetric, "Convert a vector to a skew-symmetric matrix");
    m.def("skew_symmetric_to_vector", &skewSymmetricToVector, "Convert a skew-symmetric matrix to a vector");
    m.def("lla_to_ecef", &llaToECEF, "Convert latitude, longitude, and altitude to ECEF coordinates");
    m.def("ecef_to_lla", &ecefToLLA, "Convert ECEF coordinates to latitude, longitude, and altitude");
    m.def("principal_radii", &principalRadii, "Compute the principal radii of curvature at a given latitude and longitude");
    m.def("gravity", &gravity, "Compute the gravity at a given latitude and altitude using the Somigliana method");
    m.def("gravitation", &gravitation, "Compute the gravitational force in the ECEF frame as a vector");
    m.def("rate_ned", &rateNED, "Compute the NED frame rotation rate at a given latitude");

}