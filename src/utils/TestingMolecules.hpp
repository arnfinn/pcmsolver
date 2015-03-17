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

#ifndef TESTINGMOLECULES_HPP
#define TESTINGMOLECULES_HPP

#include <iostream>
#include <vector>

#include "Config.hpp"

#include <Eigen/Dense>

#include "Atom.hpp"
#include "Molecule.hpp"
#include "PhysicalConstants.hpp"
#include "Sphere.hpp"
#include "Symmetry.hpp"

/*! Returns the ammonia molecule
 */
Molecule NH3();

/*! Returns the CH3+ molecule
 */
Molecule CH3();

/*! Returns the C2H4 molecule
 */
Molecule C2H4();

/*! Returns the benzene molecule
 */
Molecule C6H6();

/*! Returns a single dummy atom, centered at the origin and with the given radius
 */
template <int group>
Molecule dummy(double radius = 1.0)
{
    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    std::vector<Sphere> spheres;
    Sphere sph1(origin,  radius);
    spheres.push_back(sph1);

    enum pointGroup { pgC1, pgC2, pgCs, pgCi, pgD2, pgC2v, pgC2h, pgD2h };
    Symmetry pGroup;
    switch(group) {
    case(pgC1):
        pGroup = buildGroup(0, 0, 0, 0);
        break;
    case(pgC2):
        // C2 as generated by C2z
        pGroup = buildGroup(1, 3, 0, 0);
        break;
    case(pgCs):
        // Cs as generated by Oyz
        pGroup = buildGroup(1, 1, 0, 0);
        break;
    case(pgCi):
        // Ci as generated by i
        pGroup = buildGroup(1, 7, 0, 0);
        break;
    case(pgD2):
        // D2 as generated by C2z and C2x
        pGroup = buildGroup(2, 3, 6, 0);
        break;
    case(pgC2v):
        // C2v as generated by Oyz and Oxz
        pGroup = buildGroup(2, 1, 2, 0);
        break;
    case(pgC2h):
        // C2h as generated by Oxy and i
        pGroup = buildGroup(2, 4, 7, 0);
        break;
    case(pgD2h):
        // D2h as generated by Oxy, Oxz and Oyz
        pGroup = buildGroup(3, 4, 2, 1);
        break;
    default:
        pGroup = buildGroup(0, 0, 0, 0);
        break;
    }

    Molecule dummy(spheres);
    dummy.pointGroup(pGroup);

    return dummy;
};

Molecule NH3()
{
    int nAtoms = 4;

    Eigen::Vector3d N( -0.000000000,   -0.104038047,    0.000000000);
    Eigen::Vector3d H1(-0.901584415,    0.481847022,   -1.561590016);
    Eigen::Vector3d H2(-0.901584415,    0.481847022,    1.561590016);
    Eigen::Vector3d H3( 1.803168833,    0.481847022,    0.000000000);

    Eigen::MatrixXd geom(3, nAtoms);
    geom.col(0) = N.transpose();
    geom.col(1) = H1.transpose();
    geom.col(2) = H2.transpose();
    geom.col(3) = H3.transpose();
    Eigen::Vector4d charges, masses;
    charges << 7.0, 1.0, 1.0, 1.0;
    masses  << 14.0030740, 1.0078250, 1.0078250, 1.0078250;
    std::vector<Atom> atoms;
    atoms.push_back( Atom("Nitrogen", "N", charges(0), masses(0), 2.929075493, N,
                          1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(1), masses(1), 2.267671349, H1,
                          1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(2), masses(2), 2.267671349, H2,
                          1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(3), masses(3), 2.267671349, H3,
                          1.0) );

    std::vector<Sphere> spheres;
    Sphere sph1(N,  2.929075493);
    Sphere sph2(H1, 2.267671349);
    Sphere sph3(H2, 2.267671349);
    Sphere sph4(H3, 2.267671349);
    spheres.push_back(sph1);
    spheres.push_back(sph2);
    spheres.push_back(sph3);
    spheres.push_back(sph4);

    // C1
    Symmetry pGroup = buildGroup(0, 0, 0, 0);

    return Molecule(nAtoms, charges, masses, geom, atoms, spheres, pGroup);
};

/*! Returns the H3+ molecule
 */
template <int group>
Molecule H3()
{
    int nAtoms = 3;

    Eigen::Vector3d H1( 0.735000, 0.000000, -1.333333);
    Eigen::Vector3d H2(-0.735000, 0.000000, -1.333333);
    Eigen::Vector3d H3( 0.000000, 0.000000,  2.666667);

    Eigen::MatrixXd geom(3, nAtoms);
    geom.col(0) = H1.transpose();
    geom.col(1) = H2.transpose();
    geom.col(2) = H3.transpose();
    Eigen::Vector3d charges, masses;
    charges << 1.0, 1.0, 1.0;
    masses  << 1.0078250, 1.0078250, 1.0078250;

    std::vector<Atom> atoms;
    double radiusH = (1.20 * 1.20) / convertBohrToAngstrom;
    atoms.push_back( Atom("Hydrogen", "H", charges(0), masses(0), radiusH, H1, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(1), masses(1), radiusH, H2, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(2), masses(2), radiusH, H3, 1.0) );

    std::vector<Sphere> spheres;
    Sphere sph2(H1, radiusH);
    Sphere sph3(H2, radiusH);
    Sphere sph4(H3, radiusH);
    spheres.push_back(sph2);
    spheres.push_back(sph3);
    spheres.push_back(sph4);

    enum pointGroup { pgC1, pgC2, pgCs, pgCi, pgD2, pgC2v, pgC2h, pgD2h };
    Symmetry pGroup;
    switch(group) {
    case(pgC1):
        pGroup = buildGroup(0, 0, 0, 0);
        break;
    case(pgC2v):
        // C2v as generated by Oyz and Oxz
        pGroup = buildGroup(2, 1, 2, 0);
        break;
    default:
        pGroup = buildGroup(0, 0, 0, 0);
        break;
    }

    return Molecule(nAtoms, charges, masses, geom, atoms, spheres, pGroup);
};

/*! Returns the CO2 molecule
 */
template <int group>
Molecule CO2()
{
    int nAtoms = 3;

    Eigen::Vector3d C1( 0.0000000000, 0.0000000000, 0.0000000000);
    Eigen::Vector3d O1( 2.1316110791, 0.0000000000, 0.0000000000);
    Eigen::Vector3d O2(-2.1316110791, 0.0000000000, 0.0000000000);

    Eigen::MatrixXd geom(3, nAtoms);
    geom.col(0) = C1.transpose();
    geom.col(1) = O1.transpose();
    geom.col(2) = O2.transpose();
    Eigen::Vector3d charges, masses;
    charges << 6.0, 8.0, 8.0;
    masses  << 12.00, 15.9949150, 15.9949150;

    std::vector<Atom> atoms;
    double radiusC = (1.70 * 1.20) / convertBohrToAngstrom;
    double radiusO = (1.52 * 1.20) / convertBohrToAngstrom;
    atoms.push_back( Atom("Carbon", "C", charges(0), masses(0), radiusC, C1, 1.0) );
    atoms.push_back( Atom("Oxygen", "O", charges(1), masses(1), radiusO, O1, 1.0) );
    atoms.push_back( Atom("Oxygen", "O", charges(2), masses(2), radiusO, O2, 1.0) );

    std::vector<Sphere> spheres;
    Sphere sph1(C1, radiusC);
    Sphere sph2(O1, radiusO);
    Sphere sph3(O2, radiusO);
    spheres.push_back(sph1);
    spheres.push_back(sph2);
    spheres.push_back(sph3);

    enum pointGroup { pgC1, pgC2, pgCs, pgCi, pgD2, pgC2v, pgC2h, pgD2h };
    Symmetry pGroup;
    switch(group) {
    case(pgC1):
        pGroup = buildGroup(0, 0, 0, 0);
        break;
    case(pgC2):
        // C2 as generated by C2z
        pGroup = buildGroup(1, 3, 0, 0);
        break;
    case(pgCs):
        // Cs as generated by Oyz
        pGroup = buildGroup(1, 1, 0, 0);
        break;
    case(pgCi):
        // Ci as generated by i
        pGroup = buildGroup(1, 7, 0, 0);
        break;
    case(pgD2):
        // D2 as generated by C2z and C2x
        pGroup = buildGroup(2, 3, 6, 0);
        break;
    case(pgC2v):
        // C2v as generated by Oyz and Oxz
        pGroup = buildGroup(2, 1, 2, 0);
        break;
    case(pgC2h):
        // C2h as generated by Oxy and i
        pGroup = buildGroup(2, 4, 7, 0);
        break;
    case(pgD2h):
        // D2h as generated by Oxy, Oxz and Oyz
        pGroup = buildGroup(3, 4, 2, 1);
        break;
    default:
        pGroup = buildGroup(0, 0, 0, 0);
        break;
    }

    return Molecule(nAtoms, charges, masses, geom, atoms, spheres, pGroup);
};

Molecule CH3()
{
    int nAtoms = 4;

    Eigen::Vector3d C1( 0.0006122714,  0.0000000000,  0.0000000000);
    Eigen::Vector3d H1( 1.5162556382, -1.3708721537,  0.0000000000);
    Eigen::Vector3d H2(-0.7584339548,  0.6854360769,  1.7695110698);
    Eigen::Vector3d H3(-0.7584339548,  0.6854360769, -1.7695110698);

    Eigen::MatrixXd geom(3, nAtoms);
    geom.col(0) = C1.transpose();
    geom.col(1) = H1.transpose();
    geom.col(2) = H2.transpose();
    geom.col(3) = H3.transpose();
    Eigen::Vector4d charges, masses;
    charges << 6.0, 1.0, 1.0, 1.0;
    masses  << 12.00, 1.0078250, 1.0078250, 1.0078250;

    double radiusC = (1.70 * 1.20) / convertBohrToAngstrom;
    double radiusH = (1.20 * 1.20) / convertBohrToAngstrom;
    std::vector<Atom> atoms;
    atoms.push_back( Atom("Carbon",   "C", charges(0), masses(0), radiusC, C1, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(1), masses(1), radiusH, H1, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(2), masses(2), radiusH, H2, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(3), masses(3), radiusH, H3, 1.0) );

    std::vector<Sphere> spheres;
    Sphere sph1(C1, radiusC);
    Sphere sph2(H1, radiusH);
    Sphere sph3(H2, radiusH);
    Sphere sph4(H3, radiusH);
    spheres.push_back(sph1);
    spheres.push_back(sph2);
    spheres.push_back(sph3);
    spheres.push_back(sph4);

    // Cs as generated by Oxy
    Symmetry pGroup = buildGroup(1, 4, 0, 0);

    return Molecule(nAtoms, charges, masses, geom, atoms, spheres, pGroup);
};

Molecule C2H4()
{
    int nAtoms = 6;

    Eigen::Vector3d C1(0.0000000000,  0.0000000000,  1.2578920000);
    Eigen::Vector3d H1(0.0000000000,  1.7454620000,  2.3427160000);
    Eigen::Vector3d H2(0.0000000000, -1.7454620000,  2.3427160000);
    Eigen::Vector3d C2(0.0000000000,  0.0000000000, -1.2578920000);
    Eigen::Vector3d H3(0.0000000000,  1.7454620000, -2.3427160000);
    Eigen::Vector3d H4(0.0000000000, -1.7454620000, -2.3427160000);

    Eigen::MatrixXd geom(3, nAtoms);
    geom.col(0) = C1.transpose();
    geom.col(2) = H1.transpose();
    geom.col(3) = H2.transpose();
    geom.col(1) = C2.transpose();
    geom.col(4) = H3.transpose();
    geom.col(5) = H4.transpose();
    Eigen::VectorXd charges(6), masses(6);
    charges << 6.0, 1.0, 1.0, 6.0, 1.0, 1.0;
    masses  << 12.00, 1.0078250, 1.0078250, 12.0, 1.0078250, 1.0078250;

    double radiusC = (1.70 * 1.20) / convertBohrToAngstrom;
    double radiusH = (1.20 * 1.20) / convertBohrToAngstrom;
    std::vector<Atom> atoms;
    atoms.push_back( Atom("Carbon",   "C", charges(0), masses(0), radiusC, C1, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(2), masses(2), radiusH, H1, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(3), masses(3), radiusH, H2, 1.0) );
    atoms.push_back( Atom("Carbon",   "C", charges(1), masses(1), radiusC, C2, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(4), masses(4), radiusH, H3, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(5), masses(5), radiusH, H4, 1.0) );

    std::vector<Sphere> spheres;
    Sphere sph1(C1, radiusC);
    Sphere sph2(H1, radiusH);
    Sphere sph3(H2, radiusH);
    Sphere sph4(C2, radiusC);
    Sphere sph5(H3, radiusH);
    Sphere sph6(H4, radiusH);
    spheres.push_back(sph1);
    spheres.push_back(sph2);
    spheres.push_back(sph3);
    spheres.push_back(sph4);
    spheres.push_back(sph5);
    spheres.push_back(sph6);

    // D2h as generated by Oxy, Oxz, Oyz
    Symmetry pGroup = buildGroup(3, 4, 2, 1);

    return Molecule(nAtoms, charges, masses, geom, atoms, spheres, pGroup);
};

Molecule C6H6()
{
    int nAtoms = 12;

    // These are in Angstrom
    Eigen::Vector3d C1(5.274,  1.999, -8.568);
    Eigen::Vector3d C2(6.627,  2.018, -8.209);
    Eigen::Vector3d C3(7.366,  0.829, -8.202);
    Eigen::Vector3d C4(6.752, -0.379, -8.554);
    Eigen::Vector3d C5(5.399, -0.398, -8.912);
    Eigen::Vector3d C6(4.660,  0.791, -8.919);
    Eigen::Vector3d H1(4.704,  2.916, -8.573);
    Eigen::Vector3d H2(7.101,  2.950, -7.938);
    Eigen::Vector3d H3(8.410,  0.844, -7.926);
    Eigen::Vector3d H4(7.322, -1.296, -8.548);
    Eigen::Vector3d H5(4.925, -1.330, -9.183);
    Eigen::Vector3d H6(3.616,  0.776, -9.196);
    // Scale
    C1 /= convertBohrToAngstrom;
    C2 /= convertBohrToAngstrom;
    C3 /= convertBohrToAngstrom;
    C4 /= convertBohrToAngstrom;
    C5 /= convertBohrToAngstrom;
    C6 /= convertBohrToAngstrom;
    H1 /= convertBohrToAngstrom;
    H2 /= convertBohrToAngstrom;
    H3 /= convertBohrToAngstrom;
    H4 /= convertBohrToAngstrom;
    H5 /= convertBohrToAngstrom;
    H6 /= convertBohrToAngstrom;

    Eigen::MatrixXd geom(3, nAtoms);
    geom.col(0) = C1.transpose();
    geom.col(1) = C2.transpose();
    geom.col(2) = C3.transpose();
    geom.col(3) = C4.transpose();
    geom.col(4) = C5.transpose();
    geom.col(5) = C6.transpose();
    geom.col(6) = H1.transpose();
    geom.col(7) = H2.transpose();
    geom.col(8) = H3.transpose();
    geom.col(9) = H4.transpose();
    geom.col(10) = H5.transpose();
    geom.col(11) = H6.transpose();
    Eigen::VectorXd charges(12), masses(12);
    charges << 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    masses  << 12.00, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0078250, 1.0078250, 1.0078250,
            1.0078250, 1.0078250, 1.0078250;

    double radiusC = 1.70 / convertBohrToAngstrom;
    double radiusH = 1.20 / convertBohrToAngstrom;
    std::vector<Atom> atoms;
    atoms.push_back( Atom("Carbon",   "C",  charges(0), masses(0), radiusC, C1, 1.0) );
    atoms.push_back( Atom("Carbon",   "C",  charges(1), masses(1), radiusC, C2, 1.0) );
    atoms.push_back( Atom("Carbon",   "C",  charges(2), masses(2), radiusC, C3, 1.0) );
    atoms.push_back( Atom("Carbon",   "C",  charges(3), masses(3), radiusC, C4, 1.0) );
    atoms.push_back( Atom("Carbon",   "C",  charges(4), masses(4), radiusC, C5, 1.0) );
    atoms.push_back( Atom("Carbon",   "C",  charges(5), masses(5), radiusC, C6, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H",  charges(6), masses(6), radiusH, H1, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H",  charges(7), masses(7), radiusH, H2, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H",  charges(8), masses(8), radiusH, H3, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H",  charges(9), masses(9), radiusH, H4, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(10), masses(10), radiusH, H5, 1.0) );
    atoms.push_back( Atom("Hydrogen", "H", charges(11), masses(11), radiusH, H6, 1.0) );

    std::vector<Sphere> spheres;
    Sphere sph1(C1, radiusC);
    Sphere sph2(C2, radiusC);
    Sphere sph3(C3, radiusC);
    Sphere sph4(C4, radiusC);
    Sphere sph5(C5, radiusC);
    Sphere sph6(C6, radiusC);

    Sphere sph7(H1, radiusH);
    Sphere sph8(H2, radiusH);
    Sphere sph9(H3, radiusH);
    Sphere sph10(H4, radiusH);
    Sphere sph11(H5, radiusH);
    Sphere sph12(H6, radiusH);

    spheres.push_back(sph1);
    spheres.push_back(sph2);
    spheres.push_back(sph3);
    spheres.push_back(sph4);
    spheres.push_back(sph5);
    spheres.push_back(sph6);
    spheres.push_back(sph7);
    spheres.push_back(sph8);
    spheres.push_back(sph9);
    spheres.push_back(sph10);
    spheres.push_back(sph11);
    spheres.push_back(sph12);

    // D2h as generated by Oxy, Oxz, Oyz
    Symmetry pGroup = buildGroup(0, 0, 0, 0);

    return Molecule(nAtoms, charges, masses, geom, atoms, spheres, pGroup);
};

#endif // TESTINGMOLECULES_HPP
