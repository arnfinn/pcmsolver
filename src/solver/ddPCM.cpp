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

#include "ddPCM.hpp"

#include <cmath>

#include "utils/Molecule.hpp"

namespace pcm {
namespace solver {
ddPCM::ddPCM(const Molecule & m) : nSpheres_(m.spheres().size()), molecule_(m) {
  int ncav = 0;
  int ngrid = 110;
  Lmax_ = 6;
  nBasis_ = (Lmax_ + 1) * (Lmax_ + 1);
  int iconv = 7;
  int igrad = 0;
  int iprint = 2;
  int nproc = 1;
  double eps = 78.39;
  double eta = 0.1;

  nSpheres_ = m.spheres().size();
  double * xs = new double[nSpheres_];
  double * ys = new double[nSpheres_];
  double * zs = new double[nSpheres_];
  double * rs = new double[nSpheres_];
  for (int i = 0; i < nSpheres_; ++i) {
    xs[i] = m.spheres(i).center(0);
    ys[i] = m.spheres(i).center(1);
    zs[i] = m.spheres(i).center(2);
    rs[i] = m.spheres(i).radius;
  }
  ddinit(&iprint,
         &nproc,
         &Lmax_,
         &ngrid,
         &iconv,
         &igrad,
         &eps,
         &eta,
         &nSpheres_,
         xs,
         ys,
         zs,
         rs,
         &ncav);
  cavity_ = Eigen::Matrix3Xd::Zero(3, ncav);
  copy_cavity(cavity_.data());
  delete[] xs;
  delete[] ys;
  delete[] zs;
  delete[] rs;
}

ddPCM::~ddPCM() { memfree(); }

Eigen::MatrixXd ddPCM::computeX(const Psi & psi, const Eigen::VectorXd & phi) const {
  double Es = 0.0;
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(nBasis_, nSpheres_);
  itsolv_direct(phi.data(), psi().data(), X.data(), &Es);

  /* Test computation of xi */
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nBasis_, nSpheres_);
  int nll = int(cavity_.cols()/nSpheres_); // Number of Lebedev-Laikov grid points
  Eigen::MatrixXd xi = Eigen::MatrixXd::Zero(nSpheres_, nll);
  itsolv_adjoint(psi().data(), S.data());
  compute_xi(S.data(), xi.data());

  return X;
}

Psi::Psi() : nBasis_(0), nSpheres_(0) {}

Psi::Psi(int nBasis, int nSpheres, const Eigen::VectorXd & charge)
    : nBasis_(nBasis), nSpheres_(nSpheres), PsiDiscrete_(Eigen::MatrixXd::Zero(nBasis, nSpheres)) {

  PCMSOLVER_ASSERT(nSpheres_ == charge.size());
  for (int i = 0; i < charge.size(); ++i) {
    PsiDiscrete_(0, i) = std::sqrt(4.0 * M_PI) * charge(i);
  }
}

Eigen::MatrixXd Psi::operator()(const BeckeGrid & grid,
                                const Eigen::VectorXd & weightRho) const {
  PCMSOLVER_ASSERT(grid.cols() == weightRho.size());

  Eigen::MatrixXd PsiContinuous = Eigen::MatrixXd::Zero(nBasis_, nSpheres_);

  return PsiDiscrete_ + PsiContinuous;
}
} // namespace solver
} // namespace pcm
