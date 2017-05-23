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
  int lmax = 6;
  int iconv = 7;
  int igrad = 0;
  int iprint = 2;
  int nproc = 1;
  double eps = 78.39;
  double eta = 0.1;

  Psi_ = detail::Psi((lmax+1)*(lmax+1), nSpheres_, m.charges());

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
         &lmax,
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

Eigen::MatrixXd ddPCM::computeX(const Eigen::VectorXd & phi) const {
  int nbasis = 7 * 7;
  double Es = 0.0;
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(nbasis, nSpheres_);
  itsolv_direct(phi.data(), Psi_().data(), X.data(), &Es);

  /* Test computation of xi */
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nbasis, nSpheres_);
  int nll = int(cavity_.cols()/nSpheres_);
  Eigen::MatrixXd xi = Eigen::MatrixXd::Zero(nSpheres_, nll);
  itsolv_adjoint(Psi_().data(), S.data());
  compute_xi(S.data(), xi.data());

  return X;
}

namespace detail{
Psi::Psi(int nBasis, int nSpheres, const Eigen::VectorXd & charge)
    : PsiDiscrete_(Eigen::MatrixXd::Zero(nBasis, nSpheres)) {

  PCMSOLVER_ASSERT(nSpheres == charge.size());
  for (int i = 0; i < charge.size(); ++i) {
    PsiDiscrete_(0, i) = std::sqrt(4.0 * M_PI) * charge(i);
  }
}

Eigen::MatrixXd Psi::operator()(int nBasis,
                                int nSpheres,
                                const BeckeGrid & grid,
                                const Eigen::VectorXd & weightRho) const {
  PCMSOLVER_ASSERT(grid.cols() == weightRho.size());

  Eigen::MatrixXd PsiContinuous = Eigen::MatrixXd::Zero(nBasis, nSpheres);

  return PsiDiscrete_ + PsiContinuous;
}
} // namespace detail
} // namespace solver
} // namespace pcm
