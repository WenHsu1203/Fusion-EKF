#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // update the state by using Kalman Filter equations
  VectorXd y = z - (H_*x_);
  MatrixXd S = H_*P_*H_.transpose() + R_;
  MatrixXd K = P_*H_.transpose()*S.inverse();

  x_ = x_ + (K*y);
  MatrixXd I = MatrixXd::Identity(x_.size(),x_.size());
  P_ = (I-K*H_)*P_;
}

VectorXd cartesian2polar(const VectorXd& v)
{
  VectorXd polar(3);
  double px = v(0);
  double py = v(1);
  double vx = v(2);
  double vy = v(3);

  double rho = sqrt(px*px + py*py);
  double phi = atan2(py,px);
  double threshold = 0.0001;
  double d_rho;
  if(rho > threshold)
    d_rho = (px*vx + py*vy)/rho;
  else
    d_rho = 0.0;
  polar<< rho, phi, d_rho;
  return polar;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // update the state by using Extended Kalman Filter equations
  VectorXd z_pred = cartesian2polar(x_);
  VectorXd y = z - z_pred;
  // Normalized the phi such that the value within [-2pi, 2pi]
  y(1) = atan2(sin(y(1)), cos(y(1)));

  MatrixXd S = H_*P_*H_.transpose() + R_;
  MatrixXd K = P_*H_.transpose()*S.inverse();

  x_ = x_ + (K*y);
  MatrixXd I = MatrixXd::Identity(x_.size(),x_.size());
  P_ = (I-K*H_)*P_;
}
