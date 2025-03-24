/// Strapdown navigation toolbox for various navigation filters
///
/// This crate provides a set of tools for implementing navigation filters in Rust. The filters are implemented as structs that can be initialized and updated with new sensor data. The filters are designed to be used in a strapdown navigation system, where the orientation of the sensor is known and the sensor data can be used to estimate the position and velocity of the sensor.
/// Primarily built off of three crate dependencies:
/// - nav-types: Provides the basic coordinate types and conversions.
/// - nalgebra: Provides the linear algebra tools for the filters.
/// - haversine-rs: Provides the haversine formula for calculating distances between two points on the Earth's surface.
/// All other functionality is built on top of these crates. The primary reference text is _Prinicples of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition_ by Paul D. Groves. Where applicable, calculations will be referenced by the appropriate equation number tied to the book. In general, variables will be named according to the quantity they represent and not the symbol used in the book. For example, the Earth's radius will be named `EARTH_RADIUS` instead of `a`.
pub mod earth;
pub mod filter;
pub mod strapdown;


pub fn wrap_to_180<T>(angle: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
{
    let mut wrapped: T = angle;
    while wrapped > T::from(180.0) {
        wrapped -= T::from(360.0);
    }
    while wrapped < T::from(-180.0) {
        wrapped += T::from(360.0);
    }
    return wrapped;
}
pub fn wrap_to_360<T>(angle: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
{
    let mut wrapped: T = angle;
    while wrapped > T::from(360.0) {
        wrapped -= T::from(360.0);
    }
    while wrapped < T::from(0.0) {
        wrapped += T::from(360.0);
    }
    return wrapped;
}

pub fn wrap_to_pi<T>(angle: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
{
    let mut wrapped: T = angle;
    while wrapped > T::from(std::f64::consts::PI) {
        wrapped -= T::from(2.0 * std::f64::consts::PI);
    }
    while wrapped < T::from(-std::f64::consts::PI) {
        wrapped += T::from(2.0 * std::f64::consts::PI);
    }
    return wrapped;
}
pub fn wrap_to_2pi<T>(angle: T) -> T
where
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<f64>,
    T: PartialOrd + Copy + std::ops::SubAssign + std::ops::AddAssign + From<i32>,
{
    let mut wrapped: T = angle;
    while wrapped > T::from(2.0 * std::f64::consts::PI) {
        wrapped -= T::from(2.0 * std::f64::consts::PI);
    }
    while wrapped < T::from(0.0) {
        wrapped += T::from(2.0 * std::f64::consts::PI);
    }
    return wrapped;
}

#[cfg(test)]

mod tests {
    #[test]
    fn test_wrap_to_180() {
        assert_eq!(super::wrap_to_180(190.0), -170.0);
        assert_eq!(super::wrap_to_180(-190.0), 170.0);
        assert_eq!(super::wrap_to_180(0.0), 0.0);
        assert_eq!(super::wrap_to_180(180.0), 180.0);
        assert_eq!(super::wrap_to_180(-180.0), -180.0);
    }
    #[test]
    fn test_wrap_to_360() {
        assert_eq!(super::wrap_to_360(370.0), 10.0);
        assert_eq!(super::wrap_to_360(-10.0), 350.0);
        assert_eq!(super::wrap_to_360(0.0), 0.0);
    }
    #[test]
    fn test_wrap_to_pi() {
        assert_eq!(super::wrap_to_pi(3.0 * std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_pi(-3.0 * std::f64::consts::PI), -std::f64::consts::PI);
        assert_eq!(super::wrap_to_pi(0.0), 0.0);
        assert_eq!(super::wrap_to_pi(std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_pi(-std::f64::consts::PI), -std::f64::consts::PI);
    }
    #[test]
    fn test_wrap_to_2pi() {
        assert_eq!(super::wrap_to_2pi(7.0 * std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_2pi(-5.0 * std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_2pi(0.0), 0.0);
        assert_eq!(super::wrap_to_2pi(std::f64::consts::PI), std::f64::consts::PI);
        assert_eq!(super::wrap_to_2pi(-std::f64::consts::PI), std::f64::consts::PI);
    }
}