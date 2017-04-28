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

#include "Config.hpp"

#include "FCMangle.hpp"

#include <Eigen/Core>

#include "utils/Molecule.hpp"

namespace pcm {
namespace solver {
/*! \file ddPCM.hpp
 *  \class ddPCM
 *  \brief Class wrapper for ddPCM.
 *  \author Roberto Di Remigio
 *  \date 2017
 *
 *  We use the Non-Virtual Interface idiom.
 */
class ddPCM {
public:
  ddPCM(const Molecule & m);
  ~ddPCM();
  Eigen::Matrix3Xd cavity() const { return cavity_; }
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> computeCharges(Eigen::VectorXd phi);  
  //subroutine itsolv_direct(phi,psi,sigma,ene)

private:
  Eigen::Matrix3Xd cavity_;
    int nspheres;
};

#define ddinit FortranCInterface_MODULE(ddcosmo, ddinit, DDCOSMO, DDINIT)
extern "C" void ddinit(int * n,
                       double * x,
                       double * y,
                       double * z,
                       double * rvdw,
                       int * ncav);

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

#define itsolv_direct FortranCInterface_MODULE(ddcosmo, itsolv_direct, DDCOSMO, ITSOLV_DIRECT)
extern "C" void itsolv_direct(double * phi,
			      double * psi,
			      double * sigma,
			      double * ene);

#define copy_cavity                                                                 \
  FortranCInterface_MODULE(ddcosmo, copy_cavity, DDCOSMO, COPY_CAVITY)
extern "C" void copy_cavity(double * cavity);
} // namespace solver
} // namespace pcm

#endif // DDPCM_HPP
