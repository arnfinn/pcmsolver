#ifndef NUMERICALINTEGRATOR_HPP
#define NUMERICALINTEGRATOR_HPP

#include <cmath>
#include <functional>
#include <iosfwd>
#include <stdexcept>
#include <vector>

#include "Config.hpp"

#include <Eigen/Dense>

#include "IntegratorHelperFunctions.hpp"
#include "Element.hpp"
#include "AnisotropicLiquid.hpp"
#include "SphericalDiffuse.hpp"
#include "IonicLiquid.hpp"
#include "UniformDielectric.hpp"
#include "Vacuum.hpp"

/*! \file NumericalIntegrator.hpp
 *  \struct NumericalIntegrator
 *  \brief Implementation of the single and double layer operators matrix representation using one-point collocation
 *  \author Roberto Di Remigio
 *  \date 2015
 *
 *  Calculates the diagonal elements of S and D by collocation, using numerical integration.
 */

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

struct NumericalIntegrator
{
    /**@{ Single and double layer potentials for a Vacuum Green's function by collocation: numerical integration of diagonal */
    template <typename DerivativeTraits>
    Eigen::MatrixXd singleLayer(const Vacuum<DerivativeTraits, NumericalIntegrator> & gf, const std::vector<Element> & e) const {
        auto kernelS = std::bind(&Vacuum<DerivativeTraits, NumericalIntegrator>::kernelS, gf, _1, _2);
        auto diagonal = [kernelS] (const Element & el) -> double { return integrator::integrateS<32, 16>(kernelS, el); };
        return integrator::singleLayer(e, diagonal, kernelS);
    }
    template <typename DerivativeTraits>
    Eigen::MatrixXd doubleLayer(const Vacuum<DerivativeTraits, NumericalIntegrator> & gf, const std::vector<Element> & e) const {
        auto kernelD = std::bind(&Vacuum<DerivativeTraits, NumericalIntegrator>::kernelD, gf, _1, _2, _3);
        auto diagonal = [kernelD] (const Element & el) -> double { return integrator::integrateD<32, 16>(kernelD, el); };
        return integrator::doubleLayer(e, diagonal, kernelD);
    }
    /**@}*/

    /**@{ Single and double layer potentials for a UniformDielectric Green's function by collocation: numerical integration of diagonal */
    template <typename DerivativeTraits>
    Eigen::MatrixXd singleLayer(const UniformDielectric<DerivativeTraits, NumericalIntegrator> & gf, const std::vector<Element> & e) const {
        auto kernelS = std::bind(&UniformDielectric<DerivativeTraits, NumericalIntegrator>::kernelS, gf, _1, _2);
        auto diagonal = [kernelS] (const Element & el) -> double { return integrator::integrateS<32, 16>(kernelS, el); };
        return integrator::singleLayer(e, diagonal, kernelS);
    }
    template <typename DerivativeTraits>
    Eigen::MatrixXd doubleLayer(const UniformDielectric<DerivativeTraits, NumericalIntegrator> & gf, const std::vector<Element> & e) const {
        auto kernelD = std::bind(&UniformDielectric<DerivativeTraits, NumericalIntegrator>::kernelD, gf, _1, _2, _3);
        auto diagonal = [kernelD] (const Element & el) -> double { return integrator::integrateD<32, 16>(kernelD, el); };
        return integrator::doubleLayer(e, diagonal, kernelD);
    }
    /**@}*/

    /**@{ Single and double layer potentials for a IonicLiquid Green's function by collocation: numerical integration of diagonal */
    template <typename DerivativeTraits>
    Eigen::MatrixXd singleLayer(const IonicLiquid<DerivativeTraits, NumericalIntegrator> & gf, const std::vector<Element> & e) const {
        auto kernelS = std::bind(&IonicLiquid<DerivativeTraits, NumericalIntegrator>::kernelS, gf, _1, _2);
        auto diagonal = [kernelS] (const Element & el) -> double { return integrator::integrateS<32, 16>(kernelS, el); };
        return integrator::singleLayer(e, diagonal, kernelS);
    }
    template <typename DerivativeTraits>
    Eigen::MatrixXd doubleLayer(const IonicLiquid<DerivativeTraits, NumericalIntegrator> & gf, const std::vector<Element> & e) const {
        auto kernelD = std::bind(&IonicLiquid<DerivativeTraits, NumericalIntegrator>::kernelD, gf, _1, _2, _3);
        auto diagonal = [kernelD] (const Element & el) -> double { return integrator::integrateD<32, 16>(kernelD, el); };
        return integrator::doubleLayer(e, diagonal, kernelD);
    }
    /**@}*/

    /**@{ Single and double layer potentials for a AnisotropicLiquid Green's function by collocation: numerical integration of diagonal */
    template <typename DerivativeTraits>
    Eigen::MatrixXd singleLayer(const AnisotropicLiquid<DerivativeTraits, NumericalIntegrator> & gf, const std::vector<Element> & e) const {
        auto kernelS = std::bind(&AnisotropicLiquid<DerivativeTraits, NumericalIntegrator>::kernelS, gf, _1, _2);
        auto diagonal = [kernelS] (const Element & el) -> double { return integrator::integrateS<32, 16>(kernelS, el); };
        return integrator::singleLayer(e, diagonal, kernelS);
    }
    template <typename DerivativeTraits>
    Eigen::MatrixXd doubleLayer(const AnisotropicLiquid<DerivativeTraits, NumericalIntegrator> & gf, const std::vector<Element> & e) const {
        auto kernelD = std::bind(&AnisotropicLiquid<DerivativeTraits, NumericalIntegrator>::kernelD, gf, _1, _2, _3);
        auto diagonal = [kernelD] (const Element & el) -> double { return integrator::integrateD<32, 16>(kernelD, el); };
        return integrator::doubleLayer(e, diagonal, kernelD);
    }
    /**@}*/

    /**@{ Single and double layer potentials for a SphericalDiffuse Green's function by collocation: numerical integration of diagonal */
    template <typename ProfilePolicy>
    Eigen::MatrixXd singleLayer(const SphericalDiffuse<NumericalIntegrator, ProfilePolicy> & /* gf */, const std::vector<Element> & e) const {
//      // The singular part is "integrated" as usual, while the nonsingular part is evaluated in full
//      double area = e.area();
//      // Diagonal of S inside the cavity
//      double Sii_I = factor_ * std::sqrt(4 * M_PI / area);
//      // "Diagonal" of Coulomb singularity separation coefficient
//      double coulomb_coeff = gf.coefficientCoulomb(e.center(), e.center());
//      // "Diagonal" of the image Green's function
//      double image = gf.imagePotential(e.center(), e.center());

//      return (Sii_I / coulomb_coeff + image);
        return Eigen::MatrixXd::Zero(e.size(), e.size());
    }
    template <typename ProfilePolicy>
    Eigen::MatrixXd doubleLayer(const SphericalDiffuse<NumericalIntegrator, ProfilePolicy> & /* gf */, const std::vector<Element> & e) const {
//      // The singular part is "integrated" as usual, while the nonsingular part is evaluated in full
//      double area = e.area();
//      double radius = e.sphere().radius();
//      // Diagonal of S inside the cavity
//      double Sii_I = factor_ * std::sqrt(4 * M_PI / area);
//      // Diagonal of D inside the cavity
//      double Dii_I = -factor_ * std::sqrt(M_PI/ area) * (1.0 / radius);
//      // "Diagonal" of Coulomb singularity separation coefficient
//      double coulomb_coeff = gf.coefficientCoulomb(e.center(), e.center());
//      // "Diagonal" of the directional derivative of the Coulomb singularity separation coefficient
//      double coeff_grad = gf.coefficientCoulombDerivative(e.normal(), e.center(), e.center()) / std::pow(coulomb_coeff, 2);
//      // "Diagonal" of the directional derivative of the image Green's function
//      double image_grad = gf.imagePotentialDerivative(e.normal(), e.center(), e.center());

//      double eps_r2 = 0.0;
//      std::tie(eps_r2, std::ignore) = gf.epsilon(e.center());

//      return eps_r2 * (Dii_I / coulomb_coeff - Sii_I * coeff_grad + image_grad);
        return Eigen::MatrixXd::Zero(e.size(), e.size());
    }
    /**@}*/
};

#endif // NUMERICALINTEGRATOR_HPP
