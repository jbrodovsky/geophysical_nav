pub mod earth;
pub mod strapdown;

use nav_types::{ECEF, WGS84};
use nalgebra::{Vector3, Matrix3};
use strapdown::{StrapdownState, IMUData};

fn main() {
    println!("Equitorial gravity: {:?}", earth::gravity(&0.0, &0.0));
    let gravity = earth::gravitation(&0.0, &0.0, &0.0);
    println!("Equitorial gravity vector: {:?}", &gravity);
    println!("Polar gravity: {:?}", earth::gravity(&90.0, &0.0));
    let gravity = earth::gravitation(&90.0, &0.0, &0.0);
    println!("Polar gravity vector: {:?}", &gravity);

    let wgs84: WGS84<f64> = WGS84::from_degrees_and_meters(0.0, 0.0, 0.0);
    let ecef: ECEF<f64> = ECEF::from(wgs84);
    let ecef_vec: Vector3<f64> = Vector3::new(ecef.x(), ecef.y(), ecef.z());
    let omega_ie: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::RATE_VECTOR);
    let prod = &omega_ie * &omega_ie * &ecef_vec;
    let rot = earth::ecef_to_lla(&0.0, &0.0);
    //gravity + omega_ie * omega_ie * ecef_vec
    println!("Omega_ie: {:?}", &omega_ie);
    println!("ECEF: {:?}", &ecef_vec);
    println!("ECEF centrifugal: {:?}", prod);
    println!("LLA centrifugal: {:?}", rot * prod);
    
    println!("==============================================");
    println!("Freefall test");
    let g = earth::gravity(&0.0, &0.0);
    let imu = IMUData { accel: Vector3::new(0.0, 0.0, -&g), gyro: Vector3::new(0.0, 0.0, 0.0) };
    let mut state = StrapdownState::new();
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    
    println!("==============================================");
    println!("Northward test");
    let imu = IMUData { accel: Vector3::new(1.0, 0.0, 0.0), gyro: Vector3::new(0.0, 0.0, 0.0) };
    let mut state = StrapdownState::new();
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);

    println!("==============================================");
    println!("Eastward holonomic test");
    let imu = IMUData { accel: Vector3::new(0.0, 1.0, 0.0), gyro: Vector3::new(0.0, 0.0, 0.0) };
    let mut state = StrapdownState::new();
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);

    println!("==============================================");
    println!("Eastward heading test");
    let imu = IMUData { accel: Vector3::new(1.0, 0.0, 0.0), gyro: Vector3::new(0.0, 0.0, 0.0) };
    let mut state = StrapdownState::new_from(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 90.0),
    );
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
    state.forward(&imu, 1.0);
    println!("{:?}", &state);
}
