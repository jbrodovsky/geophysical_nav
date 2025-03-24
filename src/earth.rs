//! Earth-related constants and functions
//!
//! This module contains constants and functions related to the Earth's shape and gravity.
//! The Earth is modeled as an ellipsoid (WGS84) with a semi-major axis and a semi-minor
//! axis. The Earth's gravity is modeled as a function of the latitude and altitude using
//! the Somingliana method. The Earth's rotation rate is also included in this module.
//! This module relies on the `nav-types` crate for the coordinate types and conversions,
//! but provides additonal functionality for calculating rotations for the strapdown
//! navigation filters. This permits the transformation of additional quantities (velocity,
//! acceleration, etc.) between the Earth-centered Earth-fixed (ECEF) frame and the
//! local-level frame.
//!
//! # Coordinate Systems
//! The WGS84 ellipsoidal model is the primary model used for the Earth's shape. This crate
//! is primarily concerned with the ECEF and local-level frames, in addition to the basic
//! body frame of the vehicle. The ECEF frame is a right-handed Cartesian coordinate system
//! with the origin at the Earth's center. The local-level frame is a right-handed Cartesian
//! coordinate system with the origin at the sensor's position. The local-level frame is
//! defined by the tangent to the ellipsoidal surface at the sensor's position. The body
//! frame is a right-handed Cartesian coordinate system with the origin at the sensor's
//! center of mass. The body frame is defined by the sensor's orientation.
//!
//! For basic positional conversions, the `nav-types` crate is used. This crate provides
//! the `WGS84` and `ECEF` types for representing the Earth's position in geodetic and
//! Cartesian coordinates, respectively. The `nav-types` crate also provides the necessary
//! conversions between the two coordinate systems.
//!
//! # Rotation Functions
//! The rotations needed for the strapdown navigation filters are not directly supported
//! by the `nav-types` crate. These functions provide the necessary rotations that are
//! primarily used for projecting the velocity and acceleration vectors. The rotations
//! are primarily used to convert between the ECEF and local-level frames.

// ----------
// Working notes:
// The canonical strapdown navigation state vector is WGS84 geodetic position (latitude, longitude, altitude) and local tangent plane (NED) velocities (north, east,
// down). The state vector is updated by integrating the IMU measurements (body frame) to estimate the position and velocity of the sensor. Velocities are updated
// in NED, whereas the positions are updated in WGS84.
// ----------
// Rotations can be handled using nalgebra's Rotation3 type, which can be converted to a DCM using the into() method. The Rotation3 type can be created from
// Euler angles for the body to local-level frame rotation. The inverse of the Rotation3 type can be used to convert from the local-level frame to the body frame.
// ----------
use ::nalgebra::{Matrix3, Vector3};
use ::nav_types::{ECEF, WGS84};
use ::std::f64::consts::PI;

// Earth constants (WGS84)
pub const RATE: f64 = 7.2921159e-5; // rad/s (omega_ie)
pub const RATE_VECTOR: Vector3<f64> = Vector3::new(0.0, 0.0, RATE);
pub const EQUATORIAL_RADIUS: f64 = 6378137.0; // meters
pub const POLAR_RADIUS: f64 = 6356752.31425; // meters
pub const ECCENTRICITY: f64 = 0.0818191908425; // unitless
pub const ECCENTRICITY_SQUARED: f64 = ECCENTRICITY * ECCENTRICITY;
pub const GE: f64 = 9.7803253359; // m/s^2, equitorial radius
pub const GP: f64 = 9.8321849378; // m/s^2, polar radius
pub const F: f64 = 1.0 / 298.257223563; // Flattening factor
pub const K: f64 = (POLAR_RADIUS * GP - EQUATORIAL_RADIUS * GE) / (EQUATORIAL_RADIUS * GE); // Somingliana's constant
// Angle conversions
const DEG2RAD: f64 = PI / 180.0;
// const RAD2DEG: f64 = 180.0 / PI;
// Geodetic conversions
// const DH2RS: f64 = DEG2RAD / 3600.0; // Degrees (lat/lon) per second to radians per seconds
// const RS2DH: f64 = 1.0 / DH2RS; // Radians per second to degrees per second
// const DRH2RRS: f64 = DEG2RAD / 60.0; // Degrees per root-hour to radians per root-second  

// Generic rotation functions
//pub fn rpy_to_dcm(roll: &f64, pitch: &f64, yaw: &f64) -> Matrix3<f64> {
//    return Rotation3::from_euler_angles(*roll, *pitch, *yaw).into();
//}

//pub fn rotate_llf_to_body(roll: &f64, pitch: &f64, yaw: &f64) -> Matrix3<f64> {
//    return Rotation3::from_euler_angles(*roll, *pitch, *yaw).into();
//}
//pub fn rotate_body_to_llf(roll: &f64, pitch: &f64, yaw: &f64) -> Matrix3<f64> {
//    return Rotation3::from_euler_angles(*roll, *pitch, *yaw).inverse().into();
//}

/// Convert a three-element vector to a skew-symmetric matrix
/// Groves' notation uses a lot of skew-symmetric matrices to represent cross products
/// and to perform more concise matrix operations (particularly invovling rotations).
/// This function converts a three-element vector to a skew-symmetric matrix.
///
/// # Skew-symetric matrix conversion
///
/// Give a nalgebra vector `v` = [v1, v2, v3], the skew-symmetric matrix `skew` is defined as:
///
/// ```text
/// skew = |  0  -v3   v2 |
///        | v3   0   -v1 |
///        |-v2   v1   0  |
/// ```
///
/// # Example
/// ```
/// use nalgebra::{Vector3, Matrix3};
///
/// let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
/// let skew: Matrix3<f64> = earth::vector_to_skew_symmetric(&v);
/// ```
pub fn vector_to_skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    let mut skew: Matrix3<f64> = Matrix3::zeros();
    skew[(0, 1)] = -v[2];
    skew[(0, 2)] = v[1];
    skew[(1, 0)] = v[2];
    skew[(1, 2)] = -v[0];
    skew[(2, 0)] = -v[1];
    skew[(2, 1)] = v[0];
    return skew;
}
/// Convert a skew-symmetric matrix to a three-element vector
/// This function converts a skew-symmetric matrix to a three-element vector. This is the
/// inverse operation of the `vector_to_skew_symmetric` function.
///
/// /// # Skew-symetric matrix conversion
///
/// Give a nalgebra Matrix3 `skew` where
/// ```text
/// skew = |  0  -v3   v2 |
///        | v3   0   -v1 |
///        |-v2   v1   0  |
/// ```
/// the vector `v` is defined as v = [v1, v2, v3]
///
/// # Example
/// ```
/// use nalgebra::{Vector3, Matrix3};
/// let skew: Matrix3<f64> = Matrix3::new(0.0, -3.0, 2.0, 3.0, 0.0, -1.0, -2.0, 1.0, 0.0);
/// let v: Vector3<f64> = earth::skew_symmetric_to_vector(&skew);///
/// ```
pub fn skew_symmetric_to_vector(skew: &Matrix3<f64>) -> Vector3<f64> {
    let mut v: Vector3<f64> = Vector3::zeros();
    v[0] = skew[(2, 1)];
    v[1] = skew[(0, 2)];
    v[2] = skew[(1, 0)];
    return v;
}
// --- Earth coordinate roation functions ------------------------------------------------------
// `nav-types` can handle the basic positional coordinate conversions, but the rotations needed
// for the strapdown navigation filters are not directly supported. These functions provide the
// necessary rotations that are primarily used for projecting the velocity and acceleration vectors.
pub fn eci_to_ecef(time: f64) -> Matrix3<f64> {
    let mut rot: Matrix3<f64> = Matrix3::zeros();
    rot[(0, 0)] = (RATE * time).cos();
    rot[(0, 1)] = (RATE * time).sin();
    rot[(1, 0)] = -(RATE * time).sin();
    rot[(1, 1)] = (RATE * time).cos();
    rot[(2, 2)] = 1.0;
    return rot;
}
pub fn ecef_to_eci(time: f64) -> Matrix3<f64> {
    return eci_to_ecef(time).transpose();
}
pub fn ecef_to_lla(latitude: &f64, longitude: &f64) -> Matrix3<f64> {
    let lat: f64 = (*latitude).to_radians();
    let lon: f64 = (*longitude).to_radians();

    let mut rot: Matrix3<f64> = Matrix3::zeros();
    rot[(0, 0)] = -lon.sin() * lat.cos();
    rot[(0, 1)] = -lon.sin() * lat.sin();
    rot[(0, 2)] = lat.cos();
    rot[(1, 0)] = -lon.sin();
    rot[(1, 1)] = lon.cos();
    rot[(2, 0)] = -lat.cos() * lon.cos();
    rot[(2, 1)] = -lat.cos() * lon.sin();
    rot[(2, 2)] = -lat.sin();
    return rot;
}
pub fn lla_to_ecef(latitude: &f64, longitude: &f64) -> Matrix3<f64> {
    return ecef_to_lla(latitude, longitude).transpose();
}

/// Convert a local-level frame to a body frame
pub fn lla_to_body(roll: &f64, pitch: &f64, yaw: &f64) -> Matrix3<f64> {
    let roll_rad: f64 = (*roll).to_radians();
    let pitch_rad: f64 = (*pitch).to_radians();
    let yaw_rad: f64 = (*yaw).to_radians() + PI / 2.0;

    let mut rot: Matrix3<f64> = Matrix3::zeros();
    rot[(0, 0)] = yaw_rad.cos() * pitch_rad.cos();
    rot[(0, 1)] = yaw_rad.cos() * pitch_rad.sin() * roll_rad.sin() - yaw_rad.sin() * roll_rad.cos();
    rot[(0, 2)] = yaw_rad.cos() * pitch_rad.sin() * roll_rad.cos() + yaw_rad.sin() * roll_rad.sin();
    rot[(1, 0)] = yaw_rad.sin() * pitch_rad.cos();
    rot[(1, 1)] = yaw_rad.sin() * pitch_rad.sin() * roll_rad.sin() + yaw_rad.cos() * roll_rad.cos();
    rot[(1, 2)] = yaw_rad.sin() * pitch_rad.sin() * roll_rad.cos() - yaw_rad.cos() * roll_rad.sin();
    rot[(2, 0)] = -pitch_rad.sin();
    rot[(2, 1)] = pitch_rad.cos() * roll_rad.sin();
    rot[(2, 2)] = pitch_rad.cos() * roll_rad.cos();
    return rot;
}
/// Convert a body frame to a local-level frame
pub fn body_to_lla(roll: &f64, pitch: &f64, yaw: &f64) -> Matrix3<f64> {
    return lla_to_body(roll, pitch, yaw).transpose();
}

/// ## Earth properties
/// The Earth is modeled as an ellipsoid with a semi-major axis and a semi-minor axis. The Earth's
/// gravity is modeled as a function of the latitude and altitude using the Somingliana method.
/// The Earth's rotation rate is also included in this module.

// calculate principal radii of curvature
pub fn principal_radii(latitude: &f64, altitude: &f64) -> (f64, f64, f64) {
    let latitude_rad: f64 = (latitude).to_radians();
    let sin_lat: f64 = latitude_rad.sin();
    let sin_lat_sq: f64 = sin_lat * sin_lat;
    let r_n: f64 = (EQUATORIAL_RADIUS * (1.0 - ECCENTRICITY_SQUARED))
        / (1.0 - ECCENTRICITY_SQUARED * sin_lat_sq).powf(3.0 / 2.0);
    let r_e: f64 = EQUATORIAL_RADIUS / (1.0 - ECCENTRICITY_SQUARED * sin_lat_sq).sqrt();
    let r_p: f64 = r_e * latitude_rad.cos() + altitude;
    return (r_n, r_e, r_p);
}
pub fn gravity(latitude: &f64, altitude: &f64) -> f64 {
    let sin_lat: f64 = (latitude).to_radians().sin();
    let g0: f64 = (GE * (1.0 + K * sin_lat * sin_lat))
        / (1.0 - ECCENTRICITY_SQUARED * sin_lat * sin_lat).sqrt();
    return g0 - 3.08e-6 * altitude;
}

/// Calculate the gravitational force vector in the LLA frame
///
pub fn gravitation(latitude: &f64, altitude: &f64) -> Vector3<f64> {
    let wgs84: WGS84<f64> = WGS84::from_degrees_and_meters(*latitude, 0.0, *altitude);
    let ecef: ECEF<f64> = ECEF::from(wgs84);
    // Get centrifugal terms in ECEF
    let ecef_vec: Vector3<f64> = Vector3::new(ecef.x(), ecef.y(), ecef.z());
    let omega_ie: Matrix3<f64> = vector_to_skew_symmetric(&RATE_VECTOR);
    // Get rotation and gravity in LLA    
    let rot: Matrix3<f64> = ecef_to_lla(latitude, &0.0);
    let gravity: Vector3<f64> = Vector3::new(0.0, 0.0, gravity(latitude, altitude));
    // Calculate the effective gravity vector combining gravity and centrifugal terms    
    return gravity + rot * omega_ie * omega_ie * ecef_vec;
}

/// Calculate the Earth rotation rate vector in the local-level frame
pub fn earth_rate_lla(latitude: &f64) -> Vector3<f64> {
    let sin_lat: f64 = (latitude).to_radians().sin();
    let cos_lat: f64 = (latitude).to_radians().cos();
    let omega_ie: Vector3<f64> = Vector3::new(RATE * cos_lat, 0.0, -RATE * sin_lat);
    return omega_ie;
}
pub fn transport_rate(latitude: &f64, altitude: &f64, velocities: &Vector3<f64>) -> Vector3<f64> {
    let (r_n, r_e, _) = principal_radii(latitude, altitude);
    let lat_rad = latitude * DEG2RAD;
    let omega_en_n: Vector3<f64> = Vector3::new(
        -velocities[1] / (r_n + *altitude),
        velocities[0] / (r_e + *altitude),
        velocities[0] * lat_rad.tan() / (r_n + *altitude),
    );
    return omega_en_n;
}

// === Unit tests ===
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    #[test]
    fn test_vector_to_skew_symmetric() {
        let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
        let skew: Matrix3<f64> = vector_to_skew_symmetric(&v);
        assert_eq!(skew[(0, 1)], -v[2]);
        assert_eq!(skew[(0, 2)], v[1]);
        assert_eq!(skew[(1, 0)], v[2]);
        assert_eq!(skew[(1, 2)], -v[0]);
        assert_eq!(skew[(2, 0)], -v[1]);
        assert_eq!(skew[(2, 1)], v[0]);
    }
    #[test]
    fn test_skew_symmetric_to_vector() {
        let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
        let skew: Matrix3<f64> = vector_to_skew_symmetric(&v);
        let v2: Vector3<f64> = skew_symmetric_to_vector(&skew);
        assert_eq!(v, v2);
    }
    #[test]
    fn test_gravity() {
        // test polar gravity
        let latitude: f64 = 90.0;
        let grav = gravity(&latitude, &0.0);
        assert_approx_eq!(grav, GP);
        // test equatorial gravity
        let latitude: f64 = 0.0;
        let grav = gravity(&latitude, &0.0);
        assert_approx_eq!(grav, GE);
    }
    #[test]
    fn test_gravitation() {
        // test equatorial gravity
        let latitude: f64 = 0.0;
        let altitude: f64 = 0.0;
        let grav: Vector3<f64> = gravitation(&latitude, &altitude);
        assert_approx_eq!(grav[0], 0.0);
        assert_approx_eq!(grav[1], 0.0);
        assert_approx_eq!(grav[2], GE + 0.0339, 1e-4);
        // test polar gravity
        let latitude: f64 = 90.0;
        let grav: Vector3<f64> = gravitation(&latitude, &altitude);
        assert_approx_eq!(grav[0], 0.0);
        assert_approx_eq!(grav[1], 0.0);
        assert_approx_eq!(grav[2], GP);
    }
}
