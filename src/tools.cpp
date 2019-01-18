#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Initializing the RMSE vector:
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  unsigned int est_size = estimations.size();

  // Check the validity of the inputs:
  //  - Estimation vector size shouldn't be 0.
  //  - Estimation vector size should be equal to ground truth vector size.
  if ( est_size == 0 || est_size != ground_truth.size() ) {
     cout << "Invalid estimation or ground_truth data!" << endl;
     return rmse;
  }

  // Accumulate squared residuals:
  for (unsigned int i = 0; i < est_size; ++i) {
     VectorXd residual = estimations[i] - ground_truth[i];
     residual = residual.array() * residual.array();
     rmse += residual;
  }

  // Calculate the mean:
  rmse /= est_size;

  // Calculate the squared root:
  rmse = rmse.array().sqrt();

  return rmse;
}