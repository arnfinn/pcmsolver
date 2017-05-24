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

#ifndef DDPCM_HPP
#define DDPCM_HPP

#include <iosfwd>
#include <vector>

#include "Config.hpp"

#include "FCMangle.hpp"

#include <Eigen/Core>

#include "utils/Sphere.hpp"
#include "utils/Molecule.hpp"

namespace pcm {
namespace solver {

class Psi;

/*! \file ddPCM.hpp
 *  \class ddPCM
 *  \brief Class wrapper for ddPCM.
 *  \author Roberto Di Remigio
 *  \date 2017
 *
 *  We use the Non-Virtual Interface idiom.
 */

typedef Eigen::Matrix<double, 4, Eigen::Dynamic> BeckeGrid;

class ddPCM {
public:
  ddPCM(const Molecule & m);
  ~ddPCM();
  Eigen::Matrix3Xd cavity() const { return cavity_; }
  Eigen::MatrixXd computeX(const Psi & psi, const Eigen::VectorXd & phi) const;

  /*! \brief Compute the intermediate \f$ \eta_n^j \f$ required for the formation of \f$ \mathbf{F}^{s,2} \f$
   *  \param[in] grid Becke grid of points
   *  \param[in] X coefficients of the multipolar representation of the local ASC
   *  \return The \f$\eta_n^j\f$ indexed over the Becke points
   *  \note It is assumed that the Becke points are clustered according
   *  to sorting employed for atoms/spheres, thus enabling the use of only
   *  one index for this intermediate:
   *  \f[
   *    \eta_n^j = \sum_{lm} \frac{4\pi}{2l+1}
   *  \f]
   *  The Fock matrix contribution is the contraction of this intermediate with
   *  the overlap distribution coefficients sampled at the Becke points:
   *  \f[
   *    F_{\mu\nu}^{s,2} = \left\langle \frac{\pderiv \Psi}{\pderiv D_{\mu\nu}}, X\right\rangle =
   *    \sum_{j=1}^M \sum_{n=1}^{N_\mathrm{B}}
   *    \tau_n^j\Omega_{\mu\nu}(\mathbf{x}_n^j)\eta_n^j
   *  \f]
   */
  Eigen::MatrixXd computeEta(const BeckeGrid & grid,
                             const Eigen::MatrixXd & X);
  /*! \brief Compute the intermediate \f$ \xi_n^j \f$ required for the formation of \f$ \mathbf{F}^{s,1} \f$
   *  \return The \f$\xi_n^j\f$ indexed over the Lebedev-Laikov points
   *  \note This requires no input from outside, since it is calculated entirely Fortran-side:
   *  \f[
   *    \xi_n^j = w_nU_n^j\sum_{lm}[S_j]_l^mY_l^m(\mathbf{y}_n)
   *  \f]
   *  The Fock matrix contribution is the contraction of this intermediate with
   *  the eletrostatic potential integrals at the Ledebev-Laikov points:
   *  \f[
   *    F_{\mu\nu}^{s,1} = \left\langle S, \frac{\pderiv \g}{\pderiv D_{\mu\nu}}\right\rangle =
   *    \sum_{j=1}^M \sum_{n=1}^{N_\mathrm{LL}} V_{n, \mu\nu}^j\xi_n^j
   *  \f]
   *
   */
  Eigen::MatrixXd computeXi();
  int nBasis() const { return nBasis_; }
  int nSpheres() const { return nSpheres_; }

private:
  int Lmax_;
  int nBasis_;
  int nSpheres_;
  Molecule molecule_;
  Eigen::Matrix3Xd cavity_;
};

// TODO Extend to the case of general multipoles
// \warning Such an extension needs a general cartesian-to-spherical transform
class Psi {
  public:
    Psi();
    Psi(int nBasis, int nSpheres, const Eigen::VectorXd & charge);
    Eigen::MatrixXd operator()() const { return PsiDiscrete_; }
    /*! \brief Compute \f$ \Psi \f$ vector for a continous charge distribution
     *         \f$ \rho \f$ and add on top of the discrete Psi
     *  \param[in] nBasis
     *  \param[in] nSpheres
     *  \param[in] grid Becke grid of points
     *  \param[in] weightRho product of weights \f$ \tau_{j}\f$ and \f$\rho\f$ values
     *  \return the multipolar representation of the \f$\Psi\f$ vector over the
     *  spheres
     *
     *  The \f$ \Psi \f$ vector is needed in the computation of the polarization
     *energy
     *  and as the right-hand side in the adjoint ddCOSMO equation.
     *  For continuous charge distributions it is computed as:
     *  \f[
     *      [\Psi_j]_l^m = \frac{4\pi}{2l+1}\sum_{n=1}^{N_\mathrm{B}^j}
     *      \tau^{j}_n\rho(\mathbf{x}^j_n)\frac{x_<^l}{x_>^{l+1}}Y_l^m(\mathbf{s}_n^j)
     *  \f]
     */
    Eigen::MatrixXd operator()(const BeckeGrid & grid,
                               const Eigen::VectorXd & weightRho) const;

  private:
    int nBasis_;
    int nSpheres_;
    Eigen::MatrixXd PsiDiscrete_;
};

// TODO Consider injecting the Ledeved-Laikov grid inside ddCOSMO
#define ddinit FortranCInterface_MODULE(ddcosmo, ddinit, DDCOSMO, DDINIT)
extern "C" void ddinit(const int * iprint,
                       const int * nproc,
                       const int * lmax,
                       const int * ngrid,
                       const int * iconv,
                       const int * igrad,
                       const double * eps,
                       const double * eta,
                       const int * n,
                       const double * x,
                       const double * y,
                       const double * z,
                       const double * rvdw,
                       int * ncav);

#define copy_cavity                                                                 \
  FortranCInterface_MODULE(ddcosmo, copy_cavity, DDCOSMO, COPY_CAVITY)
extern "C" void copy_cavity(double * cavity);

#define itsolv_direct                                                               \
  FortranCInterface_MODULE(ddcosmo, itsolv_direct, DDCOSMO, ITSOLV_DIRECT)
extern "C" void itsolv_direct(const double * phi,
                              const double * psi,
                              double * sigma,
                              double * ene);

#define itsolv_adjoint                                                               \
  FortranCInterface_MODULE(ddcosmo, itsolv_adjoint, DDCOSMO, ITSOLV_ADJOINT)
extern "C" void itsolv_adjoint(const double * psi,
                              double * S);

#define compute_xi                                                                  \
  FortranCInterface_MODULE(ddcosmo, compute_xi, DDCOSMO, COMPUTE_XI)
extern "C" void compute_xi(const double * S,
                              double * xi);

#define compute_harmonic_extension_psi                                              \
  FortranCInterface_MODULE(ddcosmo,                                                 \
                           compute_harmonic_extension_psi,                          \
                           DDCOSMO,                                                 \
                           COMPUTE_HARMONIC_EXTENSION_PSI)
extern "C" void compute_harmonic_extension_psi(double * psi,
                                               const int * nbecke,
                                               const double * becke,
                                               const double * taurho);

#define memfree FortranCInterface_MODULE(ddcosmo, memfree, DDCOSMO, MEMFREE)
extern "C" void memfree();

#define fdoga FortranCInterface_MODULE(ddcosmo, fdoga, DDCOSMO, FDOGA)
extern "C" void fdoga(int * isph, double * xi, double * phi, double * fx);

#define fdoka FortranCInterface_MODULE(ddcosmo, fdoka, DDCOSMO, FDOKA)
extern "C" void fdoka(int * isph,
                      double * sigma,
                      double * xi,
                      double * basloc,
                      double * dbsloc,
                      double * vplm,
                      double * vcos,
                      double * vsin,
                      double * fx);

#define fdokb FortranCInterface_MODULE(ddcosmo, fdokb, DDCOSMO, FDOKB)
extern "C" void fdokb(int * isph,
                      double * sigma,
                      double * xi,
                      double * basloc,
                      double * dbsloc,
                      double * vplm,
                      double * vcos,
                      double * vsin,
                      double * fx);
} // namespace solver
} // namespace pcm

#endif // DDPCM_HPP
