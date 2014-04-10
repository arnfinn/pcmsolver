#include <iostream>

#include "Config.hpp"

#include "EigenPimpl.hpp"
#include <boost/filesystem.hpp>

#include "CPCMSolver.hpp"
#include "DerivativeTypes.hpp"
#include "GePolCavity.hpp"
#include "Symmetry.hpp"
#include "UniformDielectric.hpp"
#include "Vacuum.hpp"

// Disable obnoxious warnings from Google Test headers
#if defined (__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall" 
#pragma GCC diagnostic ignored "-Weffc++" 
#pragma GCC diagnostic ignored "-Wextra"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop
#endif

namespace fs = boost::filesystem;

TEST(IEFSolver, pointChargeGePolC1) 
{
	// Set up cavity
	Eigen::Vector3d N(0.0, 0.0, 0.0); 		
	std::vector<Sphere> spheres;      		
	Sphere sph1(N, 1.0);      		
	spheres.push_back(sph1);          		
	double area = 0.4;
	double probeRadius = 0.0;
	double minRadius = 100.0;
	// C1
	Symmetry group = buildGroup(0, 0, 0, 0);
	GePolCavity cavity(spheres, area, probeRadius, minRadius, group);
	fs::rename("PEDRA.OUT", "PEDRA.OUT.c1");
	
	double permittivity = 78.39;
	Vacuum<AD_directional> * gfInside = new Vacuum<AD_directional>(); 
	UniformDielectric<AD_directional> * gfOutside = new UniformDielectric<AD_directional>(permittivity);
	bool symm = true;
	double correction = 0.0;
	CPCMSolver solver(gfInside, gfOutside, symm, correction);
	solver.buildSystemMatrix(cavity);

	double charge = 1.0;
	int size = cavity.size();
	Eigen::VectorXd fake_mep = Eigen::VectorXd::Zero(size);
	for (int i = 0; i < size; ++i)
	{
		Eigen::Vector3d center = cavity.elementCenter(i);
		double distance = center.norm();
		fake_mep(i) = charge / distance; 
	}
	// The total ASC for a conductor is -Q
	// for CPCM it will be -Q*[(epsilon-1)/epsilon]
	Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(size);
	solver.compCharge(fake_mep, fake_asc);
	double totalASC = - charge * (permittivity - 1) / permittivity;
	double totalFakeASC = fake_asc.sum();
	std::cout << "totalASC - totalFakeASC = " << totalASC - totalFakeASC << std::endl;
	EXPECT_NEAR(totalASC, totalFakeASC, 3e-3);
}

TEST(IEFSolver, pointChargeGePolCs) 
{
	// Set up cavity
	Eigen::Vector3d N(0.0, 0.0, 0.0); 		
	std::vector<Sphere> spheres;      		
	Sphere sph1(N, 1.0);      		
	spheres.push_back(sph1);          		
	double area = 0.4;
	double probeRadius = 0.0;
	double minRadius = 100.0;
	// Cs as generated by Oyz
	Symmetry group = buildGroup(1, 1, 0, 0);
	GePolCavity cavity(spheres, area, probeRadius, minRadius, group);
	fs::rename("PEDRA.OUT", "PEDRA.OUT.cs");
	
	double permittivity = 78.39;
	Vacuum<AD_directional> * gfInside = new Vacuum<AD_directional>(); 
	UniformDielectric<AD_directional> * gfOutside = new UniformDielectric<AD_directional>(permittivity);
	bool symm = true;
	double correction = 0.0;
	CPCMSolver solver(gfInside, gfOutside, symm, correction);
	solver.buildSystemMatrix(cavity);

	double charge = 1.0;
	int irr_size = cavity.irreducible_size();
	Eigen::VectorXd fake_mep = Eigen::VectorXd::Zero(irr_size);
	// Calculate it only on the irreducible portion of the cavity
	// then replicate it according to the point group
	for (int i = 0; i < irr_size; ++i)
	{
		Eigen::Vector3d center = cavity.elementCenter(i);
		double distance = center.norm();
		fake_mep(i) = charge / distance; 
	}
	int nr_irrep = cavity.pointGroup().nrIrrep();
	// The total ASC for a conductor is -Q
	// for CPCM it will be -Q*[(epsilon-1)/epsilon]
	Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(irr_size);
	solver.compCharge(fake_mep, fake_asc);
	double totalASC = - charge * (permittivity - 1) / permittivity;
	double totalFakeASC = fake_asc.sum() * nr_irrep;
	std::cout << "totalASC - totalFakeASC = " << totalASC - totalFakeASC << std::endl;
	EXPECT_NEAR(totalASC, totalFakeASC, 3e-3);
}

TEST(IEFSolver, pointChargeGePolC2) 
{
	// Set up cavity
	Eigen::Vector3d N(0.0, 0.0, 0.0); 		
	std::vector<Sphere> spheres;      		
	Sphere sph1(N, 1.0);      		
	spheres.push_back(sph1);          		
	double area = 0.4;
	double probeRadius = 0.0;
	double minRadius = 100.0;
	// C2 as generated by C2z 
	Symmetry group = buildGroup(1, 3, 0, 0);
	GePolCavity cavity(spheres, area, probeRadius, minRadius, group);
	fs::rename("PEDRA.OUT", "PEDRA.OUT.c2");
	
	double permittivity = 78.39;
	Vacuum<AD_directional> * gfInside = new Vacuum<AD_directional>(); 
	UniformDielectric<AD_directional> * gfOutside = new UniformDielectric<AD_directional>(permittivity);
	bool symm = true;
	double correction = 0.0;
	CPCMSolver solver(gfInside, gfOutside, symm, correction);
	solver.buildSystemMatrix(cavity);

	double charge = 1.0;
	int irr_size = cavity.irreducible_size();
	Eigen::VectorXd fake_mep = Eigen::VectorXd::Zero(irr_size);
	// Calculate it only on the irreducible portion of the cavity
	// then replicate it according to the point group
	for (int i = 0; i < irr_size; ++i)
	{
		Eigen::Vector3d center = cavity.elementCenter(i);
		double distance = center.norm();
		fake_mep(i) = charge / distance; 
	}
	int nr_irrep = cavity.pointGroup().nrIrrep();
	// The total ASC for a conductor is -Q
	// for CPCM it will be -Q*[(epsilon-1)/epsilon]
	Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(irr_size);
	solver.compCharge(fake_mep, fake_asc);
	double totalASC = - charge * (permittivity - 1) / permittivity;
	double totalFakeASC = fake_asc.sum() * nr_irrep;
	std::cout << "totalASC - totalFakeASC = " << totalASC - totalFakeASC << std::endl;
	EXPECT_NEAR(totalASC, totalFakeASC, 3e-3);
}

TEST(IEFSolver, pointChargeGePolCi) 
{
	// Set up cavity
	Eigen::Vector3d N(0.0, 0.0, 0.0); 		
	std::vector<Sphere> spheres;      		
	Sphere sph1(N, 1.0);      		
	spheres.push_back(sph1);          		
	double area = 0.4;
	double probeRadius = 0.0;
	double minRadius = 100.0;
	// Ci as generated by i
	Symmetry group = buildGroup(1, 7, 0, 0);
	GePolCavity cavity(spheres, area, probeRadius, minRadius, group);
	fs::rename("PEDRA.OUT", "PEDRA.OUT.ci");
	
	double permittivity = 78.39;
	Vacuum<AD_directional> * gfInside = new Vacuum<AD_directional>(); 
	UniformDielectric<AD_directional> * gfOutside = new UniformDielectric<AD_directional>(permittivity);
	bool symm = true;
	double correction = 0.0;
	CPCMSolver solver(gfInside, gfOutside, symm, correction);
	solver.buildSystemMatrix(cavity);

	double charge = 1.0;
	int irr_size = cavity.irreducible_size();
	Eigen::VectorXd fake_mep = Eigen::VectorXd::Zero(irr_size);
	// Calculate it only on the irreducible portion of the cavity
	// then replicate it according to the point group
	for (int i = 0; i < irr_size; ++i)
	{
		Eigen::Vector3d center = cavity.elementCenter(i);
		double distance = center.norm();
		fake_mep(i) = charge / distance; 
	}
	int nr_irrep = cavity.pointGroup().nrIrrep();
	// The total ASC for a conductor is -Q
	// for CPCM it will be -Q*[(epsilon-1)/epsilon]
	Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(irr_size);
	solver.compCharge(fake_mep, fake_asc);
	double totalASC = - charge * (permittivity - 1) / permittivity;
	double totalFakeASC = fake_asc.sum() * nr_irrep;
	std::cout << "totalASC - totalFakeASC = " << totalASC - totalFakeASC << std::endl;
	EXPECT_NEAR(totalASC, totalFakeASC, 3e-3);
}

TEST(IEFSolver, pointChargeGePolC2h) 
{
	// Set up cavity
	Eigen::Vector3d N(0.0, 0.0, 0.0); 		
	std::vector<Sphere> spheres;      		
	Sphere sph1(N, 1.0);      		
	spheres.push_back(sph1);          		
	double area = 0.4;
	double probeRadius = 0.0;
	double minRadius = 100.0;
	// C2h as generated by Oxy and i
	Symmetry group = buildGroup(2, 4, 7, 0);
	GePolCavity cavity(spheres, area, probeRadius, minRadius, group);
	fs::rename("PEDRA.OUT", "PEDRA.OUT.c2h");
	
	double permittivity = 78.39;
	Vacuum<AD_directional> * gfInside = new Vacuum<AD_directional>(); 
	UniformDielectric<AD_directional> * gfOutside = new UniformDielectric<AD_directional>(permittivity);
	bool symm = true;
	double correction = 0.0;
	CPCMSolver solver(gfInside, gfOutside, symm, correction);
	solver.buildSystemMatrix(cavity);

	double charge = 1.0;
	int irr_size = cavity.irreducible_size();
	Eigen::VectorXd fake_mep = Eigen::VectorXd::Zero(irr_size);
	// Calculate it only on the irreducible portion of the cavity
	// then replicate it according to the point group
	for (int i = 0; i < irr_size; ++i)
	{
		Eigen::Vector3d center = cavity.elementCenter(i);
		double distance = center.norm();
		fake_mep(i) = charge / distance; 
	}
	int nr_irrep = cavity.pointGroup().nrIrrep();
	// The total ASC for a conductor is -Q
	// for CPCM it will be -Q*[(epsilon-1)/epsilon]
	Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(irr_size);
	solver.compCharge(fake_mep, fake_asc);
	double totalASC = - charge * (permittivity - 1) / permittivity;
	double totalFakeASC = fake_asc.sum() * nr_irrep;
	std::cout << "totalASC - totalFakeASC = " << totalASC - totalFakeASC << std::endl;
	EXPECT_NEAR(totalASC, totalFakeASC, 3e-3);
}

TEST(IEFSolver, pointChargeGePolD2) 
{
	// Set up cavity
	Eigen::Vector3d N(0.0, 0.0, 0.0); 		
	std::vector<Sphere> spheres;      		
	Sphere sph1(N, 1.0);      		
	spheres.push_back(sph1);          		
	double area = 0.4;
	double probeRadius = 0.0;
	double minRadius = 100.0;
	// D2 as generated by C2z and C2x
	Symmetry group = buildGroup(2, 3, 6, 0);
	GePolCavity cavity(spheres, area, probeRadius, minRadius, group);
	fs::rename("PEDRA.OUT", "PEDRA.OUT.d2");
	
	double permittivity = 78.39;
	Vacuum<AD_directional> * gfInside = new Vacuum<AD_directional>(); 
	UniformDielectric<AD_directional> * gfOutside = new UniformDielectric<AD_directional>(permittivity);
	bool symm = true;
	double correction = 0.0;
	CPCMSolver solver(gfInside, gfOutside, symm, correction);
	solver.buildSystemMatrix(cavity);

	double charge = 1.0;
	int irr_size = cavity.irreducible_size();
	Eigen::VectorXd fake_mep = Eigen::VectorXd::Zero(irr_size);
	// Calculate it only on the irreducible portion of the cavity
	// then replicate it according to the point group
	for (int i = 0; i < irr_size; ++i)
	{
		Eigen::Vector3d center = cavity.elementCenter(i);
		double distance = center.norm();
		fake_mep(i) = charge / distance; 
	}
	int nr_irrep = cavity.pointGroup().nrIrrep();
	// The total ASC for a conductor is -Q
	// for CPCM it will be -Q*[(epsilon-1)/epsilon]
	Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(irr_size);
	solver.compCharge(fake_mep, fake_asc);
	double totalASC = - charge * (permittivity - 1) / permittivity;
	double totalFakeASC = fake_asc.sum() * nr_irrep;
	std::cout << "totalASC - totalFakeASC = " << totalASC - totalFakeASC << std::endl;
	EXPECT_NEAR(totalASC, totalFakeASC, 3e-3);
}

TEST(IEFSolver, pointChargeGePolC2v) 
{
	// Set up cavity
	Eigen::Vector3d N(0.0, 0.0, 0.0); 		
	std::vector<Sphere> spheres;      		
	Sphere sph1(N, 1.0);      		
	spheres.push_back(sph1);          		
	double area = 0.4;
	double probeRadius = 0.0;
	double minRadius = 100.0;
	// C2v as generated by Oyz and Oxz
	Symmetry group = buildGroup(2, 1, 2, 0);
	GePolCavity cavity(spheres, area, probeRadius, minRadius, group);
	fs::rename("PEDRA.OUT", "PEDRA.OUT.c2v");
	
	double permittivity = 78.39;
	Vacuum<AD_directional> * gfInside = new Vacuum<AD_directional>(); 
	UniformDielectric<AD_directional> * gfOutside = new UniformDielectric<AD_directional>(permittivity);
	bool symm = true;
	double correction = 0.0;
	CPCMSolver solver(gfInside, gfOutside, symm, correction);
	solver.buildSystemMatrix(cavity);

	double charge = 1.0;
	int irr_size = cavity.irreducible_size();
	Eigen::VectorXd fake_mep = Eigen::VectorXd::Zero(irr_size);
	// Calculate it only on the irreducible portion of the cavity
	// then replicate it according to the point group
	for (int i = 0; i < irr_size; ++i)
	{
		Eigen::Vector3d center = cavity.elementCenter(i);
		double distance = center.norm();
		fake_mep(i) = charge / distance; 
	}
	int nr_irrep = cavity.pointGroup().nrIrrep();
	// The total ASC for a conductor is -Q
	// for CPCM it will be -Q*[(epsilon-1)/epsilon]
	Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(irr_size);
	solver.compCharge(fake_mep, fake_asc);
	double totalASC = - charge * (permittivity - 1) / permittivity;
	double totalFakeASC = fake_asc.sum() * nr_irrep;
	std::cout << "totalASC - totalFakeASC = " << totalASC - totalFakeASC << std::endl;
	EXPECT_NEAR(totalASC, totalFakeASC, 3e-3);
}

TEST(IEFSolver, pointChargeGePolD2h) 
{
	// Set up cavity
	Eigen::Vector3d N(0.0, 0.0, 0.0); 		
	std::vector<Sphere> spheres;
	Sphere sph1(N, 1.0);      		
	spheres.push_back(sph1);          		
	double area = 0.4;
	double probeRadius = 0.0;
	double minRadius = 100.0;
	// D2h as generated by Oxy, Oxz and Oyz
	Symmetry group = buildGroup(3, 4, 2, 1);
	GePolCavity cavity(spheres, area, probeRadius, minRadius, group);
	fs::rename("PEDRA.OUT", "PEDRA.OUT.d2h");
	
	double permittivity = 78.39;
	Vacuum<AD_directional> * gfInside = new Vacuum<AD_directional>(); 
	UniformDielectric<AD_directional> * gfOutside = new UniformDielectric<AD_directional>(permittivity);
	bool symm = true;
	double correction = 0.0;
	CPCMSolver solver(gfInside, gfOutside, symm, correction);
	solver.buildSystemMatrix(cavity);

	double charge = 1.0;
	int irr_size = cavity.irreducible_size();
	Eigen::VectorXd fake_mep = Eigen::VectorXd::Zero(irr_size);
	// Calculate it only on the irreducible portion of the cavity
	// then replicate it according to the point group
	for (int i = 0; i < irr_size; ++i)
	{
		Eigen::Vector3d center = cavity.elementCenter(i);
		double distance = center.norm();
		fake_mep(i) = charge / distance; 
	}
	int nr_irrep = cavity.pointGroup().nrIrrep();
	// The total ASC for a conductor is -Q
	// for CPCM it will be -Q*[(epsilon-1)/epsilon]
	Eigen::VectorXd fake_asc = Eigen::VectorXd::Zero(irr_size);
	solver.compCharge(fake_mep, fake_asc);
	double totalASC = - charge * (permittivity - 1) / permittivity;
	double totalFakeASC = fake_asc.sum() * nr_irrep;
	std::cout << "totalASC - totalFakeASC = " << totalASC - totalFakeASC << std::endl;
	EXPECT_NEAR(totalASC, totalFakeASC, 3e-3);
}
