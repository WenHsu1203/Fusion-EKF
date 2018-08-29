#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

	VectorXd rmse(4);
	rmse << 0,0,0,0;
	if (estimations.size()!=ground_truth.size() || estimations.size() == 0)
		return rmse;
	for (int i = 0; i < estimations.size(); i++)
	{
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		rmse += residual;
	}
	// Calculate the mean
	rmse = rmse/ estimations.size();
	// Calculate the squared root
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj = MatrixXd::Zero(3,4);

	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float px_2 = px*px;
	float py_2 = py*py;
	float c = px_2 + py_2;
	float c_sq = sqrt(c);
	float c_1_5 = sqrt(pow(c,3));
	// Check validity
	double threshold = 0.0001;
	if (c >= threshold)
	{
		double r11 = px / c_sq;
		double r12 = py / c_sq;
		double r21 = -py/ c;
		double r22 = px / c;
		double r31 = py*(vx*py-vy*px)/c_1_5;
		double r32 = px*(vy*px-vx*py)/c_1_5;

		Hj<< r11, r12, 0.0, 0.0,
			 r21, r22, 0.0, 0.0, 
			 r31, r32, r11, r12;
	}
	return Hj;

}











