/*
 * PCMSolver, an API for the Polarizable Continuum Model
 * Copyright (C) 2018 Roberto Di Remigio, Luca Frediani and contributors.
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

#include "catch.hpp"

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "TestingMolecules.hpp"
#include "cavity/GePolCavity.hpp"

using namespace pcm;
using cavity::GePolCavity;

SCENARIO("GePol cavity for a single sphere", "[gepol][gepol_point]") {
  GIVEN("A single sphere") {
    double area = 0.4;
    double probeRadius = 0.0;
    double minRadius = 100.0;
    WHEN("the sphere is obtained from a Molecule object") {
      Molecule point = dummy<0>();
      GePolCavity cavity(point, area, probeRadius, minRadius, "point");
      cavity.saveCavity("point.npz");

      /*! \class GePolCavity
       *  \test \b GePolCavityTest_size tests GePol cavity size for a point charge in
       * C1 symmetry without added spheres
       */
      THEN("the size of the cavity is") {
        int size = 32;
        int actualSize = cavity.size();
        REQUIRE(size == actualSize);
      }
      /*! \class GePolCavity
       *  \test \b GePolCavityTest_area tests GePol cavity surface area for a point
       * charge in C1 symmetry without added spheres
       */
      AND_THEN("the surface area of the cavity is") {
        double area = 4.0 * M_PI * pow(1.0, 2);
        double actualArea = cavity.elementArea().sum();
        REQUIRE(area == Approx(actualArea));
      }
      /*! \class GePolCavity
       *  \test \b GePolCavityTest_volume tests GePol cavity volume for a point
       * charge in C1 symmetry without added spheres
       */
      AND_THEN("the volume of the cavity is") {
        double volume = 4.0 * M_PI * pow(1.0, 3) / 3.0;
        Eigen::Matrix3Xd elementCenter = cavity.elementCenter();
        Eigen::Matrix3Xd elementNormal = cavity.elementNormal();
        double actualVolume = 0;
        for (int i = 0; i < cavity.size(); ++i) {
          actualVolume +=
              cavity.elementArea(i) * elementCenter.col(i).dot(elementNormal.col(i));
        }
        actualVolume /= 3;
        REQUIRE(volume == Approx(actualVolume));
      }
    }
  }

  GIVEN("A single sphere") {
    double area = 0.4;
    double probeRadius = 0.0;
    double minRadius = 100.0;
    WHEN("the sphere is obtained from a Sphere object") {
      Sphere sph(Eigen::Vector3d::Zero(), 1.0);
      GePolCavity cavity(sph, area, probeRadius, minRadius, "point");

      /*! \class GePolCavity
       *  \test \b GePolCavitySphereCTORTest_size tests GePol cavity size for a point
       * charge in C1 symmetry without added spheres
       */
      THEN("the size of the cavity is") {
        int size = 32;
        int actualSize = cavity.size();
        REQUIRE(size == actualSize);
      }
      /*! \class GePolCavity
       *  \test \b GePolCavitySphereCTORTest_area tests GePol cavity surface area for
       * a point charge in C1 symmetry without added spheres
       */
      AND_THEN("the surface area of the cavity is") {
        double area = 4.0 * M_PI * pow(1.0, 2);
        double actualArea = cavity.elementArea().sum();
        REQUIRE(area == Approx(actualArea));
      }
      /*! \class GePolCavity
       *  \test \b GePolCavitySphereCTORTest_volume tests GePol cavity volume for a
       * point charge in C1 symmetry without added spheres
       */
      AND_THEN("the volume of the cavity is") {
        double volume = 4.0 * M_PI * pow(1.0, 3) / 3.0;
        Eigen::Matrix3Xd elementCenter = cavity.elementCenter();
        Eigen::Matrix3Xd elementNormal = cavity.elementNormal();
        double actualVolume = 0;
        for (int i = 0; i < cavity.size(); ++i) {
          actualVolume +=
              cavity.elementArea(i) * elementCenter.col(i).dot(elementNormal.col(i));
        }
        actualVolume /= 3;
        REQUIRE(volume == Approx(actualVolume));
      }
    }
  }
}
