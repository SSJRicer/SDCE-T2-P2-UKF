#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

#define EPS 0.001

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  ////////////////////// MY ADDITIONS //////////////////////
  // Initialization flag:
  is_initialized_ = false;

  // Initializing the timestamp:
  time_us_ = 0;

  // Set state dimension:
  n_x_ = 5;

  // Set augmented dimension:
  n_aug_ = n_x_ + 2;

  // Set augmented sigma points amount:
  n_sig_aug_ = 2*n_aug_ + 1;

  // Define spreading parameter:
  lambda_ = 3 - n_aug_;
  ////////////////////// [END] MY ADDITIONS //////////////////////

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  ////////////////////// MY ADDITIONS //////////////////////  
  // Initalize the state vector x:
  x_ << 1, 1, 1, 0, 0;

  // Initalize the state covariance matrix P:
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Initialize the predicted sigma points matrix:
  Xsig_pred_ = MatrixXd(n_x_, n_sig_aug_);
  Xsig_pred_.fill(0.0);

  // Set weights vector:
  weights_ = VectorXd(n_sig_aug_);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Set radar measurement noise covariance matrix:
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;

  // Set laser measurement noise covariance matrix:
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;
  ////////////////////// [END] MY ADDITIONS //////////////////////
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      cout << "UKF's first measurement (RADAR):" << endl;

      // Extract the range, bearing & radial velocity from the 1st measurement:
      double rho =   meas_package.raw_measurements_[0];  // Range
      double phi =   meas_package.raw_measurements_[1];  // Bearing (angle)
      double rho_d = meas_package.raw_measurements_[2];  // Radial velocity

      // Polar -> Cartesian transformation:
      double x =  rho *  cos(phi);
      double y =  rho *  sin(phi);
      double vx = rho_d * cos(phi);
      double vy = rho_d * sin(phi);
      double v =  sqrt(vx*vx + vy*vy);
      double vv = atan2(vx, vy);  // ADDED AS PER SUBMISSION FEEDBACK

      x_.head(3) << x, y, v, vv, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      cout << "UKF's first measurement (LASER):" << endl;

      // Storing the laser's Px & Py + v = 0:
      x_.head(3) << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0;
    }

    // First measurement's timestamp:
    time_us_ = meas_package.timestamp_;

    // Done initializing, no need to predict/update:
    is_initialized_ = true;

    cout << x_ << endl;
    cout << "INITIALIZATION COMPLETE!" << endl;
    return;
  }

  // Compute the elapsed time (in seconds):
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);
  
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } 
  else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

  //// Print the output:
  //cout << "x_ =" << endl << x_ << endl;
  //cout << "P_ =" << endl << P_ << endl << endl;
}


void UKF::SigmaPointsAugmentation(MatrixXd* Xsig_aug_out) {
  // Create & set augmented mean state vector:
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // Create & set augmented state covariance matrix:
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) =         std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // Create augmented sigma point matrix:
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_aug_);

  // Create (augmented) square root matrix:
  MatrixXd A_aug = P_aug.llt().matrixL();
  
  // Reusable calculation:
  double lambda_n_sqrt = sqrt(lambda_ + n_aug_);

  // Set augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
      Xsig_aug.col(i + 1)          = x_aug + lambda_n_sqrt * A_aug.col(i);
      Xsig_aug.col(i + 1 + n_aug_) = x_aug - lambda_n_sqrt * A_aug.col(i);
  }
  
  // Save result:
  *Xsig_aug_out = Xsig_aug;
}

void UKF::SigmaPointsPrediction(double delta_t, MatrixXd Xsig_aug, MatrixXd* Xsig_pred_out) {
  // Create matrix with predicted sigma points as columns:
  MatrixXd Xsig_pred = MatrixXd(n_x_, n_sig_aug_);
  Xsig_pred.fill(0.0);
 
  // Make easier calculations later on:
  double delta_t_1_2 = 0.5 * delta_t * delta_t;

  // Create state change rate vector:
  VectorXd x_d(n_x_);

  // Create process noise vector:
  VectorXd x_dd(n_x_);

  // Predicting each sigma point:
  for (int i = 0; i < n_sig_aug_; i++) {
      // Extract the data for the equations:
      double v        = Xsig_aug(2, i);
      double yaw      = Xsig_aug(3, i);
      double yawd     = Xsig_aug(4, i);
      double nu_a     = Xsig_aug(5, i);
      double nu_yawdd = Xsig_aug(6, i);
      
      // Set process noise vector:
      x_dd << delta_t_1_2 * cos(yaw) * nu_a,
              delta_t_1_2 * sin(yaw) * nu_a,
              delta_t     * nu_a,
              delta_t_1_2 * nu_yawdd,
              delta_t     * nu_yawdd;
      
      // Set state change rate vector:
      if (fabs(yawd) < EPS) {  // Avoid division by zero:
          x_d.head(2) << v * cos(yaw) * delta_t,
                         v * sin(yaw) * delta_t;
      }
      else {
          x_d.head(2) << (v / yawd) * (sin(yaw + yawd * delta_t) - sin(yaw)),
                         (v / yawd) * (-cos(yaw + yawd * delta_t) + cos(yaw));
      }
      x_d(2) = 0;
      x_d(3) = yawd * delta_t;
      x_d(4) = 0;
      
      Xsig_pred.col(i) = Xsig_aug.col(i).head(n_x_) + x_d + x_dd;
  }

  // Save result:
  *Xsig_pred_out = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {
  // Create vector for predicted state:
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);

  // Create covariance matrix for prediction:
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  // Calculate predicted mean vector:
  for (int i = 0; i < n_sig_aug_; ++i) {  // Iterate over sigma points
    x += weights_(i) * Xsig_pred_.col(i);
  }

  // Calculate predicted state covariance matrix:
  for (int i = 0; i < n_sig_aug_; ++i) {  // Iterate over sigma points
    // State difference:
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    // Angle normalization:
    while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P += weights_(i) * x_diff * x_diff.transpose();
  }
  
  // Save results:
  *x_out = x;
  *P_out = P;
}

void UKF::Prediction(double delta_t) {
  //Create the augmented sigma points matrix:
  MatrixXd Xsig_aug = MatrixXd(n_x_, n_sig_aug_);
  Xsig_aug.fill(0.0);
  SigmaPointsAugmentation(&Xsig_aug);

  //Sigma points prediction:
  SigmaPointsPrediction(delta_t, Xsig_aug, &Xsig_pred_);

  //Predict the mean & covariance:
  PredictMeanAndCovariance(&x_, &P_);
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Set measurement dimension (laser can measure px, py):
  int n_z = 2;

  // Create the real measurement:
  VectorXd z = meas_package.raw_measurements_;

  // Create sigma points matrix in measurement space:
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_aug_);

  // Create mean predicted measurement vector:
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // Calculate mean predicted measurement:
  for (int i = 0; i < n_sig_aug_; ++i) {
      z_pred += weights_(i) * Zsig.col(i);
  }

  // Create innovation covariance matrix:
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  // Calculate innovation covariance matrix:
  for (int i = 0; i < n_sig_aug_; ++i) {
    // Residual/Error:
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Add laser measurement noise covariance matrix:
  S += R_laser_;

  // Create matrix for cross correlation:
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  // Calculate cross correlation matrix
  for (int i = 0; i < n_sig_aug_; ++i) {
      // State residual/error:
      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      // Angle normalization for state residual/error:
      while (x_diff(3) > M_PI)  x_diff(3) -= 2. * M_PI;
      while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
      
      // Measurement residual/error:
      VectorXd z_diff = Zsig.col(i) - z_pred;

      Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Estimation error:
  VectorXd y = z - z_pred;

  // Updating the state vector and state covariance matrix:
  x_ += K * y;
  P_ -= K * S * K.transpose();

  // NIS - Laser Update
  NIS_laser_ = y.transpose() * S.inverse() * y;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Set measurement dimension (radar can measure r, phi, and r_dot):
  int n_z = 3;

  // Create the real measurement:
  VectorXd z = meas_package.raw_measurements_;

  // Create measurement sigma points matrix:
  MatrixXd Zsig = MatrixXd(n_z, n_sig_aug_);
  Zsig.fill(0.0);

  // Transform sigma points into measurement space
  for (int i = 0; i < n_sig_aug_; ++i) {
      // Extract relevant data from prediction space:
      double px =  Xsig_pred_(0, i);
      double py =  Xsig_pred_(1, i);
      double v =   Xsig_pred_(2, i);
      double yaw = Xsig_pred_(3, i);
    
      // Avoid dividing by zero:
      if (fabs(px) < EPS) px = EPS;
      if (fabs(py) < EPS) py = EPS;

      // Cartesian -> Polar transform:
      double rad_r = sqrt(px*px + py*py);
      double rad_phi = atan2(py, px);
      double rad_rd = (px*cos(yaw)*v + py*sin(yaw)*v) / rad_r;
      
      Zsig.col(i) << rad_r, rad_phi, rad_rd;
  }

  // Create mean predicted measurement vector:
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // Calculate mean predicted measurement:
  for (int i = 0; i < n_sig_aug_; ++i) {
      z_pred += weights_(i) * Zsig.col(i);
  }
  
  // Create innovation covariance matrix:
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  // Calculate innovation covariance matrix:
  for (int i = 0; i < n_sig_aug_; ++i) {
      // Residual/Error:
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // Angle normalization:
      while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
      while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
      
      S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  // Add radar measurement noise covariance matrix:
  S += R_radar_;

  // Create matrix for cross correlation:
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  // Calculate cross correlation matrix
  for (int i = 0; i < n_sig_aug_; ++i) {
      // State residual/error:
      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      // Angle normalization for state residual/error:
      while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
      while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
      
      // Measurement residual/error:
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // Angle normalization for state residual/error:
      while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
      while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

      Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Estimation error:
  VectorXd y = z - z_pred;

  // Angle normalization 
  while (y(1) >  M_PI) y(1) -= 2. * M_PI;
  while (y(1) < -M_PI) y(1) += 2. * M_PI;
  
  // Updating the state vector and state covariance matrix:
  x_ += K * y;
  P_ -= K * S * K.transpose();

  // NIS - Radar Update
  NIS_radar_ = y.transpose() * S.inverse() * y;
}