#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "include/earth.hpp"
namespace py = pybind11;
using namespace earth;

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
    m.def("rotate_eci_to_ecef", &rotateECIToECEF, "Rotation matrix from ECI to ECEF");
    m.def("rotate_ecef_to_ned", &rotateECEFToNED, "Rotation matrix from ECEF to NED");
    m.def("rotate_ned_to_body", py::overload_cast<const double&, const double&, const double&>(&rotateNEDToBody), "Rotation matrix from NED to body");
    m.def("rotate_ned_to_body", py::overload_cast<const Eigen::Vector3d&>(&rotateNEDToBody), "Rotation matrix from NED to body");
    m.def("rotate_ecef_to_eci", &rotateECEFToECI, "Rotation matrix from ECEF to ECI");
    m.def("rotate_ned_to_ecef", &rotateNEDToECEF, "Rotation matrix from NED to ECEF");
    m.def("rotate_body_to_ned", py::overload_cast<const double&, const double&, const double&>(&rotateBodyToNED), "Rotation matrix from body to NED");
    m.def("rotate_body_to_ned", py::overload_cast<const Eigen::Vector3d&>(&rotateBodyToNED), "Rotation matrix from body to NED");
    m.def("rpy_to_rotation_matrix", py::overload_cast<const double&, const double&, const double&>(&rpyToRotationMatrix), "Convert roll, pitch, yaw to a rotation matrix");
    m.def("rpy_to_rotation_matrix", py::overload_cast<const Eigen::Vector3d&>(&rpyToRotationMatrix), "Convert roll, pitch, yaw to a rotation matrix");
    m.def("rotation_matrix_to_rpy", &rotationMatrixToRPY, "Convert a rotation matrix to roll, pitch, yaw");
    m.def("vector_to_skew_symmetric", &vectorToSkewSymmetric, "Convert a vector to a skew-symmetric matrix");
    m.def("skew_symmetric_to_vector", &skewSymmetricToVector, "Convert a skew-symmetric matrix to a vector");
    m.def("eci_to_ecef", py::overload_cast<const double&, const double&, const double&, const double&>(&eciToECEF), "Convert ECI coordinates to ECEF coordinates");
    m.def("eci_to_ecef", py::overload_cast<const Eigen::Vector3d&, const double&>(&eciToECEF), "Convert ECI coordinates to ECEF coordinates");
    m.def("eci_to_ecef", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&, const double&>(&eciToECEF), "Convert ECI coordinates to ECEF coordinates");
    m.def("ned_to_ecef", py::overload_cast<const double&, const double&, const double&>(&nedToECEF), "Convert latitude, longitude, and altitude to ECEF coordinates");
    m.def("ned_to_ecef", py::overload_cast<const Eigen::Vector3d&>(&nedToECEF), "Convert latitude, longitude, and altitude to ECEF coordinates");
    m.def("ned_to_ecef", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&>(&nedToECEF), "Convert latitude, longitude, and altitude to ECEF coordinates");
    m.def("ned_to_ecef", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&>(&nedToECEF), "Convert latitude, longitude, and altitude to ECEF coordinates");
    m.def("ecef_to_ned", py::overload_cast<const double&, const double&, const double&>(&ecefToNED), "Convert ECEF coordinates to latitude, longitude, and altitude");
    m.def("ecef_to_ned", py::overload_cast<const Eigen::Vector3d&>(&ecefToNED), "Convert ECEF coordinates to latitude, longitude, and altitude");
    m.def("ecef_to_ned", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&>(&ecefToNED), "Convert ECEF coordinates to latitude, longitude, and altitude");
    m.def("ecef_to_ned", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&>(&ecefToNED), "Convert ECEF coordinates to latitude, longitude, and altitude");
    m.def("principal_radii", &principalRadii, "Compute the principal radii of curvature at a given latitude and longitude");
    m.def("gravity", &gravity, "Compute the gravity at a given latitude and altitude using the Somigliana method");
    m.def("gravitation", py::overload_cast<const double&, const double&, const double&>(&gravitation), "Compute the gravitational force in the ECEF frame as a vector");
    m.def("gravitation", py::overload_cast<const Eigen::Vector3d&>(&gravitation), "Compute the gravitational force in the ECEF frame as a vector");
    m.def("rate_ned", &rateNED, "Compute the NED frame rotation rate at a given latitude");

}