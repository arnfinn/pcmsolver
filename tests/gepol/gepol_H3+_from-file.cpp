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

#define BOOST_TEST_MODULE GePolCavityH3+RestartTest

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <vector>
#include <cmath>

#include "Config.hpp"

#include <Eigen/Dense>

#include "GePolCavity.hpp"

struct GePolCavityH3RestartTest {
    GePolCavity cavity;
    GePolCavityH3RestartTest() { SetUp(); }
    void SetUp() {
        cavity.loadCavity("h3+.npz");
    }
};

/*! \class GePolCavity
 *  \test \b GePolCavityH3RestartTest_size tests GePol cavity size for H3+ loading the cavity from a .npz file
 */
BOOST_FIXTURE_TEST_CASE(size, GePolCavityH3RestartTest)
{
    int size = 312;
    int actualSize = cavity.size();
    BOOST_REQUIRE_EQUAL(size, actualSize);
}

/*! \class GePolCavity
 *  \test \b GePolCavityH3RestartTest_area tests GePol cavity surface area for H3+ loading the cavity from from a .npz file
 */
BOOST_FIXTURE_TEST_CASE(area, GePolCavityH3RestartTest)
{
    double area = 178.74700256125493;
    double actualArea = cavity.elementArea().sum();
    BOOST_REQUIRE_CLOSE(area, actualArea, 1.0e-10);
}

/*! \class GePolCavity
 *  \test \b GePolCavityH3RestartTest_volume tests GePol cavity volume for H3+ loading the cavity from from a .npz file
 */
BOOST_FIXTURE_TEST_CASE(volume, GePolCavityH3RestartTest)
{
    double volume = 196.4736029455637;
    Eigen::Matrix3Xd elementCenter = cavity.elementCenter();
    Eigen::Matrix3Xd elementNormal = cavity.elementNormal();
    double actualVolume = 0;
    for ( int i = 0; i < cavity.size(); ++i ) {
        actualVolume += cavity.elementArea(i) * elementCenter.col(i).dot(elementNormal.col(
                            i));
    }
    actualVolume /= 3;
    BOOST_REQUIRE_CLOSE(volume, actualVolume, 1.0e-10);
}
