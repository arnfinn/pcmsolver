/* pcmsolver_copyright_start */
/*
 *     PCMSolver, an API for the Polarizable Continuum Model
 *     Copyright (C) 2013-2015 Roberto Di Remigio, Luca Frediani and contributors
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
 *     PCMSolver API, see: <http://pcmsolver.readthedocs.org/>
 */
/* pcmsolver_copyright_end */

#include "catch.hpp"

#include <iostream>


#include <Eigen/Core>

#include "bi_operators/CollocationIntegrator.hpp"
#include "green/DerivativeTypes.hpp"
#include "cavity/GePolCavity.hpp"
#include "green/Vacuum.hpp"
#include "green/UniformDielectric.hpp"
#include "solver/IEFSolver.hpp"
#include "TestingMolecules.hpp"

SCENARIO("Test solver for the IEFPCM for a point charge outside a GePol cavity",
         "[solver][iefpcm][iefpcm_gepol-outcharge][outcharge]")
{
    GIVEN("An isotropic environment and a point charge")
    {
        double permittivity = 78.39;
        Vacuum<AD_directional, CollocationIntegrator> gfInside = Vacuum<AD_directional, CollocationIntegrator>();
        UniformDielectric<AD_directional, CollocationIntegrator> gfOutside =
            UniformDielectric<AD_directional, CollocationIntegrator>(permittivity);
        bool symm = true;

        double charge = 8.0;
        double totalASC = - charge * (permittivity - 1) / permittivity;

        /*! \class IEFSolver
         *  \test \b outchargeGePol tests IEFSolver using a point charge with a GePol cavity
         *  The point charge is outside the cavity at (0, 0, z > r)
         */
        WHEN("the point charge is located above the cavity")
        {
            Molecule point = dummy<0>(2.929075493);
            double area = 0.4;
            double probeRadius = 0.0;
            double minRadius = 100.0;
            std::vector<Sphere> spheres(1);
            Eigen::Vector3d chargePos(0.0, 0.0, 3.0);
            spheres[0] = Sphere(Eigen::Vector3d::Zero(), 1.0);
            GePolCavity cavity = GePolCavity(spheres, area, probeRadius, minRadius);

            IEFSolver solver(symm);
            solver.buildSystemMatrix(cavity, gfInside, gfOutside);

            size_t size = cavity.size();
            Eigen::VectorXd fake_mep = computeMEP(cavity.elements(), charge, chargePos);
            Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(size);
            fake_asc = solver.computeCharge(fake_mep);

            for (size_t i = 0; i < size; ++i) {
                INFO("fake_mep(" << i << ") = " << fake_mep(i));
            }
            for (size_t i = 0; i < size; ++i) {
                INFO("fake_asc(" << i << ") = " << fake_asc(i));
            }

            double totalFakeASC = fake_asc.sum();
            THEN("the apparent surface charge is")
            {
                CAPTURE(totalASC);
                CAPTURE(totalFakeASC);
                CAPTURE(totalASC - totalFakeASC);
                REQUIRE(totalASC == Approx(totalFakeASC).epsilon(1.0e-03));
            }
        }

    }
}