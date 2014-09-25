/* pcmsolver_copyright_start */
/*
 *     PCMSolver, an API for the Polarizable Continuum Model
 *     Copyright (C) 2013 Roberto Di Remigio, Luca Frediani and contributors
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
 *     PCMSolver API, see: <http://pcmsolver.github.io/pcmsolver-doc>
 */
/* pcmsolver_copyright_end */

#ifndef GREENSFUNCTION_HPP
#define GREENSFUNCTION_HPP

#include <iosfwd>

#include "Config.hpp"

#include <Eigen/Dense>

class DiagonalIntegrator;
class Element;

#include "IGreensFunction.hpp"

/*! \file GreensFunction.hpp
 *  \class GreensFunction
 *  \brief Templated interface for Green's functions
 *  \author Luca Frediani and Roberto Di Remigio
 *  \date 2012-2014
 *  \tparam T evaluation strategy for the function and its derivatives
 */

template <typename T>
class GreensFunction: public IGreensFunction
{
public:
    GreensFunction(bool uniform) : IGreensFunction(uniform), delta_(1.0e-4) {}
    GreensFunction(bool uniform, DiagonalIntegrator * diag) : IGreensFunction(uniform, diag), delta_(1.0e-4) {}
    virtual ~GreensFunction() {}
    /*!
     *  Returns value of the kernel of the \f$\mathcal{S}\f$ integral operator, i.e. the value of the 
     *  Greens's function for the pair of points p1, p2: \f$ G(\mathbf{p}_1, \mathbf{p}_2)\f$
     *
     *  \param[in] p1 first point
     *  \param[in] p2 second point
     */
    virtual double function(const Eigen::Vector3d & p1, const Eigen::Vector3d &p2) const;
    /*!
     *  Returns value of the kernel for the calculation of the \f$\mathcal{D}\f$ integral operator
     *  for the pair of points p1, p2:
     *  \f$ [\boldsymbol{\varepsilon}\nabla_{\mathbf{p_2}}G(\mathbf{p}_1, \mathbf{p}_2)]\cdot \mathbf{n}_{\mathbf{p}_2}\f$
     *  To obtain the kernel of the \f$\mathcal{D}^\dagger\f$ operator call this methods with \f$\mathbf{p}_1\f$
     *  and \f$\mathbf{p}_2\f$ exchanged and with \f$\mathbf{n}_{\mathbf{p}_2} = \mathbf{n}_{\mathbf{p}_1}\f$
     *
     *  \param[in] direction the direction
     *  \param[in]        p1 first point
     *  \param[in]        p2 second point
     */
    virtual double derivative(const Eigen::Vector3d & direction,
                              const Eigen::Vector3d & p1, const Eigen::Vector3d & p2) const = 0;
    /*!
     *  Returns value of the directional derivative of the
     *  Greens's function for the pair of points p1, p2:
     *  \f$ \nabla_{\mathbf{p_1}}G(\mathbf{p}_1, \mathbf{p}_2)\cdot \mathbf{n}_{\mathbf{p}_1}\f$
     *  Notice that this method returns the directional derivative with respect
     *  to the source point.
     *
     *  \param[in] normal_p1 the normal vector to p1
     *  \param[in]        p1 first point
     *  \param[in]        p2 second point
     */
    virtual double derivativeSource(const Eigen::Vector3d & normal_p1,
                                    const Eigen::Vector3d & p1, const Eigen::Vector3d & p2) const;
    /*!
     *  Returns value of the directional derivative of the
     *  Greens's function for the pair of points p1, p2:
     *  \f$ \nabla_{\mathbf{p_2}}G(\mathbf{p}_1, \mathbf{p}_2)\cdot \mathbf{n}_{\mathbf{p}_2}\f$
     *  Notice that this method returns the directional derivative with respect
     *  to the probe point.
     *
     *  \param[in] normal_p2 the normal vector to p1
     *  \param[in]        p1 first point
     *  \param[in]        p2 second point
     */
    virtual double derivativeProbe(const Eigen::Vector3d & normal_p2,
                                   const Eigen::Vector3d & p1, const Eigen::Vector3d & p2) const;
    /*!
     *  Returns full gradient of Greens's function for the pair of points p1, p2:
     *  \f$ \nabla_{\mathbf{p_1}}G(\mathbf{p}_1, \mathbf{p}_2)\f$
     *  Notice that this method returns the gradient with respect to the source point.
     *
     *  \param[in] p1 first point
     *  \param[in] p2 second point
     */
    virtual Eigen::Vector3d gradientSource(const Eigen::Vector3d & p1,
                                           const Eigen::Vector3d & p2) const;
    /*!
     *  Returns full gradient of Greens's function for the pair of points p1, p2:
     *  \f$ \nabla_{\mathbf{p_2}}G(\mathbf{p}_1, \mathbf{p}_2)\f$
     *  Notice that this method returns the gradient with respect to the probe point.
     *
     *  \param[in] p1 first point
     *  \param[in] p2 second point
     */
    virtual Eigen::Vector3d gradientProbe(const Eigen::Vector3d & p1,
                                          const Eigen::Vector3d & p2) const;
    /*!
     *  Returns full gradient of Greens's function for the pair of points p1, p2:
     *  \f$ \nabla_{\mathbf{p_1}}G(\mathbf{p}_1, \mathbf{p}_2)\f$
     *  Notice that this method returns the gradient with respect to the source point.
     *
     *  \param[in] gradient the gradient
     *  \param[in]       p1 first point
     *  \param[in]       p2 second point
     */
    virtual void gradientSource(Eigen::Vector3d & gradient, const Eigen::Vector3d & p1,
                                const Eigen::Vector3d & p2) const;
    /*!
     *  Returns full gradient of Greens's function for the pair of points p1, p2:
     *  \f$ \nabla_{\mathbf{p_2}}G(\mathbf{p}_1, \mathbf{p}_2)\f$
     *  Notice that this method returns the gradient with respect to the probe point.
     *
     *  \param[in] gradient the gradient
     *  \param[in]       p1 first point
     *  \param[in]       p2 second point
     */
    virtual void gradientProbe(Eigen::Vector3d & gradient, const Eigen::Vector3d & p1,
                               const Eigen::Vector3d & p2) const;
    
    /*!
     *  Calculates the diagonal elements of the S operator: \f$ S_{ii} \f$
     *  \param[in] e i-th finite element
     */
    virtual double diagonalS(const Element & e) const = 0;
    /*!
     *  Calculates the diagonal elements of the D operator: \f$ D_{ii} \f$
     *  \param[in] e i-th finite element
     */
    virtual double diagonalD(const Element & e) const = 0;

    virtual void delta(double value);
    virtual double delta() { return delta_; }
    virtual double epsilon() const = 0;

    friend std::ostream & operator<<(std::ostream & os, GreensFunction & gf) {
        return gf.printObject(os);
    }
protected:
    /*!
     *  Evaluates the Green's function given a pair of points
     *
     *  \param[in] source the source point
     *  \param[in]  probe the probe point
     */
    virtual T operator()(T * source, T * probe) const = 0;
    virtual std::ostream & printObject(std::ostream & os);
    double delta_;
};

#endif // GREENSFUNCTION_HPP
