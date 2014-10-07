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

#include "PurisimaIntegrator.hpp"

#include <cmath>
#include <stdexcept>

#include "Config.hpp"

#include <Eigen/Dense>

#include "Element.hpp"

double PurisimaIntegrator::computeS(const Vacuum<double> * gf, const Element & e) const {
	double area = e.area();
	return (factor_ * std::sqrt(4 * M_PI / area));
} 
double PurisimaIntegrator::computeS(const Vacuum<AD_directional> * gf, const Element & e) const {
	double area = e.area();
	return (factor_ * std::sqrt(4 * M_PI / area));
}
double PurisimaIntegrator::computeS(const Vacuum<AD_gradient> * gf, const Element & e) const {
	double area = e.area();
	return (factor_ * std::sqrt(4 * M_PI / area));
}
double PurisimaIntegrator::computeS(const Vacuum<AD_hessian> * gf, const Element & e) const {
	double area = e.area();
	return (factor_ * std::sqrt(4 * M_PI / area));
}

double PurisimaIntegrator::computeD(const Vacuum<double> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const Vacuum<AD_directional> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const Vacuum<AD_gradient> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const Vacuum<AD_hessian> * gf, const Element & e) const {
	return 0.0;
}

double PurisimaIntegrator::computeS(const UniformDielectric<double> * gf, const Element & e) const {
	double epsInv = 1.0 / gf->epsilon();
	double area = e.area();
	return (factor_ * std::sqrt(4 * M_PI / area) * epsInv);
}
double PurisimaIntegrator::computeS(const UniformDielectric<AD_directional> * gf, const Element & e) const {
	double epsInv = 1.0 / gf->epsilon();
	double area = e.area();
	return (factor_ * std::sqrt(4 * M_PI / area) * epsInv);
}
double PurisimaIntegrator::computeS(const UniformDielectric<AD_gradient> * gf, const Element & e) const {
	double epsInv = 1.0 / gf->epsilon();
	double area = e.area();
	return (factor_ * std::sqrt(4 * M_PI / area) * epsInv);
}
double PurisimaIntegrator::computeS(const UniformDielectric<AD_hessian> * gf, const Element & e) const {
	double epsInv = 1.0 / gf->epsilon();
	double area = e.area();
	return (factor_ * std::sqrt(4 * M_PI / area) * epsInv);
}

double PurisimaIntegrator::computeD(const UniformDielectric<double> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const UniformDielectric<AD_directional> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const UniformDielectric<AD_gradient> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const UniformDielectric<AD_hessian> * gf, const Element & e) const {
	return 0.0;
}

double PurisimaIntegrator::computeS(const IonicLiquid<double> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeS(const IonicLiquid<AD_directional> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeS(const IonicLiquid<AD_gradient> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeS(const IonicLiquid<AD_hessian> * gf, const Element & e) const {
	return 0.0;
}

double PurisimaIntegrator::computeD(const IonicLiquid<double> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const IonicLiquid<AD_directional> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const IonicLiquid<AD_gradient> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const IonicLiquid<AD_hessian> * gf, const Element & e) const {
	return 0.0;
}

double PurisimaIntegrator::computeS(const AnisotropicLiquid<double> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeS(const AnisotropicLiquid<AD_directional> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeS(const AnisotropicLiquid<AD_gradient> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeS(const AnisotropicLiquid<AD_hessian> * gf, const Element & e) const {
	return 0.0;
}

double PurisimaIntegrator::computeD(const AnisotropicLiquid<double> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const AnisotropicLiquid<AD_directional> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const AnisotropicLiquid<AD_gradient> * gf, const Element & e) const {
	return 0.0;
}
double PurisimaIntegrator::computeD(const AnisotropicLiquid<AD_hessian> * gf, const Element & e) const {
	return 0.0;
}
