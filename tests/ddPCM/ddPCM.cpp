/**
 * PCMSolver, an API for the Polarizable Continuum Model
 * Copyright (C) 2017 Roberto Di Remigio, Luca Frediani and collaborators.
 *
 * This file is part of PCMSolver.
 *
 * PCMSolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PCMSolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with PCMSolver.  If not, see <http://www.gnu.org/licenses/>.
 *
 * For information on the complete list of contributors to the
 * PCMSolver API, see: <http://pcmsolver.readthedocs.io/>
 */

#include "catch.hpp"

#include <iostream>

#include <Eigen/Core>

#include "utils/Molecule.hpp"
#include "TestingMolecules.hpp"
#include "solver/ddPCM.hpp"
#include "utils/MathUtils.hpp"

using namespace pcm;
using solver::ddPCM;
using solver::Psi;

/*! \class ddPCM
 *
TEST_CASE("ddCOSMO solver with NH3 molecule", "[ddPCM]") {
  Molecule molec = NH3();
  ddPCM solver(molec);
  Eigen::VectorXd potential = computeMEP(molec, solver.cavity());
  Eigen::MatrixXd charges = solver.computeCharges(potential);
}
 */

double rho_1s(const Eigen::Vector3d & point) {
  return (1.0/std::sqrt(2.0 * M_PI) * std::exp(-point.squaredNorm()));
}

TEST_CASE("ddCOSMO solver with point charge", "[ddPCM]") {
  Molecule molec = dummy<0>(1.0);
  ddPCM solver(molec);

  // Electrostatic potential at the cavity
  Eigen::VectorXd phi = computeMEP(solver.cavity(), 1.0);

  // Compute psi vector for the point charge
  Psi psi(solver.nBasis(), solver.nSpheres(), molec.charges());
  // Compute X for classical point charge
  Eigen::MatrixXd X = solver.computeX(psi(), phi);
  // Gauss' Theorem check
  REQUIRE(X(0,0)*2.0*std::sqrt(M_PI) == Approx(-1).epsilon(1.0e-03));

  // Read Becke grid from file
  // tmp contains grid points and weights
  Eigen::MatrixXd tmp = cnpy::custom::npy_load<double>("grid.npy");
  int nBeckePoints = tmp.cols();
  Eigen::Matrix3Xd beckeGrid = tmp.block(0, 0, 3, nBeckePoints);
  Eigen::VectorXd beckeWeight = tmp.row(3).transpose();
  // Sample density of a 1s Gaussian function on on Becke grid
  Eigen::VectorXd taurho = Eigen::VectorXd::Zero(nBeckePoints);
  for (int i = 0; i < nBeckePoints; ++i){
    taurho(i) = rho_1s(beckeGrid.col(i)) * beckeWeight(i);
  }
  // Compute X with full Psi vector
  X = solver.computeX(psi(beckeGrid, taurho), phi);

  // Compute eta (not used in test)
  Eigen::MatrixXd eta = Eigen::MatrixXd::Zero(molec.spheres().size(), nBeckePoints);
  solver::compute_eta(eta.data(), &nBeckePoints, beckeGrid.data(), X.data());
}
