/* pcmsolver_copyright_start */
/*
 *     PCMSolver, an API for the Polarizable Continuum Model
 *     Copyright (C) 2013-2016 Roberto Di Remigio, Luca Frediani and contributors
 *
 *     This file is part of PCMSolver.
 *
 *     PCMSolver is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Lesser General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     PCMSolver is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU Lesser General Public License for more details.
 *
 *     You should have received a copy of the GNU Lesser General Public License
 *     along with PCMSolver.  If not, see <http://www.gnu.org/licenses/>.
 *
 *     For information on the complete list of contributors to the
 *     PCMSolver API, see: <http://pcmsolver.readthedocs.io/>
 */
/* pcmsolver_copyright_end */

#include <cmath>

#include "Config.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "cavity/Cavity.hpp"
#include "cavity/Element.hpp"
#include "green/IGreensFunction.hpp"
#include "utils/MathUtils.hpp"

/*! \file SolverImpl.hpp
 *  \brief Functions common to all solvers
 *  \author Roberto Di Remigio
 *  \date 2015
 */

namespace solver {
/*! \brief Builds the matrix representation of the single layer operator S
 *  \param[in] cav the discretized cavity
 *  \param[in] gf  the Green's function
 *  \return the \f$\mathbf{S}\f$ matrix
 *  \note This function blocks the matrix according to the molecular point group.
 *  \warning This function does not pack the matrix into a collections of blocks.
 */
Eigen::MatrixXd computeS(const Cavity & cav, const IGreensFunction & gf);

/*! \brief Builds the matrix representation of the double layer operator D
 *  \param[in] cav the discretized cavity
 *  \param[in] gf  the Green's function
 *  \return the \f$\mathbf{D}\f$ matrix
 *  \note This function blocks the matrix according to the molecular point group.
 *  \warning This function does not pack the matrix into a collections of blocks.
 */
Eigen::MatrixXd computeD(const Cavity & cav, const IGreensFunction & gf);

/*! \brief Builds the **anisotropic** IEFPCM matrix
 *  \param[in] cav the discretized cavity
 *  \param[in] gf_i Green's function inside the cavity
 *  \param[in] gf_o Green's function outside the cavity
 *  \return the \f$ \mathbf{K} = \mathbf{T}^{-1}\mathbf{R}\mathbf{A} \f$ matrix
 *
 *  This function calculates the PCM matrix. We use the following definitions:
 *  \f[
 *     \begin{align}
 *       \mathbf{T} &=
 *      \left(2\pi\mathbf{I} -
 *\mathbf{D}_\mathrm{e}\mathbf{A}\right)\mathbf{S}_\mathrm{i}
 *      +\mathbf{S}_\mathrm{e}\left(2\pi\mathbf{I} +
 *      \mathbf{A}\mathbf{D}_\mathrm{i}^\dagger\right) \\
 *      \mathbf{R} &=
 *      \left(2\pi\mathbf{A}^{-1} - \mathbf{D}_\mathrm{e}\right) -
 *      \mathbf{S}_\mathrm{e}\mathbf{S}^{-1}_\mathrm{i}\left(2\pi\mathbf{A}^{-1}-\mathbf{D}_\mathrm{i}\right)
 *     \end{align}
 *  \f]
 *  The matrix is not symmetrized and is not symmetry packed.
 */
Eigen::MatrixXd anisotropicIEFMatrix(const Cavity & cav,
                                     const IGreensFunction & gf_i,
                                     const IGreensFunction & gf_o);

/*! \brief Builds the **isotropic** IEFPCM matrix
 *  \param[in] cav the discretized cavity
 *  \param[in] gf_i Green's function inside the cavity
 *  \param[in] epsilon permittivity outside the cavity
 *  \return the \f$ \mathbf{K} = \mathbf{T}^{-1}\mathbf{R}\mathbf{A} \f$ matrix
 *
 *  This function calculates the PCM matrix. We use the following definitions:
 *  \f[
 *     \begin{align}
 *       \mathbf{T} &=
 *      \left(2\pi\frac{\varepsilon+1}{\varepsilon-1}\mathbf{I} -
 *\mathbf{D}_\mathrm{i}\mathbf{A}\right)\mathbf{S}_\mathrm{i} \\
 *      \mathbf{R} &=
 *      \left(2\pi\mathbf{A}^{-1} - \mathbf{D}_\mathrm{i}\right)
 *     \end{align}
 *  \f]
 *  The matrix is not symmetrized and is not symmetry packed.
 */
Eigen::MatrixXd isotropicIEFMatrix(const Cavity & cav, const IGreensFunction & gf_i,
                                   double epsilon);

/*! \brief Builds the **anisotropic** \f$ \mathbf{T}_\varepsilon \f$ matrix
 *  \param[in] cav the discretized cavity
 *  \param[in] gf_i Green's function inside the cavity
 *  \param[in] gf_o Green's function outside the cavity
 *  \return the \f$ \mathbf{T}_\varepsilon \f$ matrix
 *
 *  We use the following definition:
 *  \f[
 *      \mathbf{T}_\varepsilon =
 *      \left(2\pi\mathbf{I} -
 *\mathbf{D}_\mathrm{e}\mathbf{A}\right)\mathbf{S}_\mathrm{i}
 *      +\mathbf{S}_\mathrm{e}\left(2\pi\mathbf{I} +
 *      \mathbf{A}\mathbf{D}_\mathrm{i}^\dagger\right)
 *  \f]
 *  The matrix is not symmetrized and is not symmetry packed.
 */
Eigen::MatrixXd anisotropicTEpsilon(const Cavity & cav, const IGreensFunction & gf_i,
                                    const IGreensFunction & gf_o);

/*! \brief Builds the **isotropic** \f$ \mathbf{T}_\varepsilon \f$ matrix
 *  \param[in] cav the discretized cavity
 *  \param[in] gf_i Green's function inside the cavity
 *  \param[in] epsilon permittivity outside the cavity
 *  \return the \f$ \mathbf{T}_\varepsilon \f$ matrix
 *
 *  We use the following definition:
 *  \f[
 *      \mathbf{T}_\varepsilon =
 *      \left(2\pi\frac{\varepsilon+1}{\varepsilon-1}\mathbf{I} -
 *\mathbf{D}_\mathrm{i}\mathbf{A}\right)\mathbf{S}_\mathrm{i}
 *  \f]
 *  The matrix is not symmetrized and is not symmetry packed.
 */
Eigen::MatrixXd isotropicTEpsilon(const Cavity & cav, const IGreensFunction & gf_i,
                                  double epsilon);

/*! \brief Builds the **anisotropic** \f$ \mathbf{R}_\infty \f$ matrix
 *  \param[in] cav the discretized cavity
 *  \param[in] gf_i Green's function inside the cavity
 *  \param[in] gf_o Green's function outside the cavity
 *  \return the \f$ \mathbf{R}_\infty\mathbf{A} \f$ matrix
 *
 *  We use the following definition:
 *  \f[
 *      \mathbf{R}_\infty =
 *      \left(2\pi\mathbf{A}^{-1} - \mathbf{D}_\mathrm{e}\right) -
 *      \mathbf{S}_\mathrm{e}\mathbf{S}^{-1}_\mathrm{i}\left(2\pi\mathbf{A}^{-1}-\mathbf{D}_\mathrm{i}\right)
 *  \f]
 *  The matrix is not symmetrized and is not symmetry packed.
 */
Eigen::MatrixXd anisotropicRinfinity(const Cavity & cav,
                                     const IGreensFunction & gf_i,
                                     const IGreensFunction & gf_o);

/*! \brief Builds the **isotropic** \f$ \mathbf{R}_\infty \f$ matrix
 *  \param[in] cav the discretized cavity
 *  \param[in] gf_i Green's function inside the cavity
 *  \return the \f$ \mathbf{R}_\infty\mathbf{A} \f$ matrix
 *
 *  We use the following definition:
 *  \f[
 *      \mathbf{R}_\infty =
 *      \left(2\pi\mathbf{A}^{-1} - \mathbf{D}_\mathrm{i}\right)
 *  \f]
 *  The matrix is not symmetrized and is not symmetry packed.
 */
Eigen::MatrixXd isotropicRinfinity(const Cavity & cav, const IGreensFunction & gf_i);
} // namespace solver
