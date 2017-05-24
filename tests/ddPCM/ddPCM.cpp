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

TEST_CASE("ddCOSMO solver with point charge", "[ddPCM]") {
  Molecule molec = dummy<0>(1.0);
  ddPCM solver(molec);
  Psi psi(solver.nBasis(), solver.nSpheres(), molec.charges());
  // Read Becke grid from file
  Eigen::MatrixXd becke = cnpy::custom::npy_load<double>("grid.npy");
  // Compute Psi vector on Becke grid
  Eigen::VectorXd taurho = Eigen::VectorXd::Zero(becke.cols());
  psi(molec.spheres(), becke, taurho);
  Eigen::VectorXd potential = computeMEP(solver.cavity(), 1.0);
  Eigen::MatrixXd X = solver.computeX(psi, potential);
  REQUIRE(X(0,0)*2.0*std::sqrt(M_PI) == Approx(-1).epsilon(1.0e-03));
}
