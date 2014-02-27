#include <vector>
#include <cmath>

#include "Config.hpp"

// Disable obnoxious warnings from Eigen headers
#if defined (__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall" 
#pragma GCC diagnostic ignored "-Weffc++" 
#pragma GCC diagnostic ignored "-Wextra"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#elif (__INTEL_COMPILER)
#pragma warning push
#pragma warning disable "-Wall"
#include <Eigen/Dense>
#pragma warning pop
#endif

#include <boost/filesystem.hpp>

#include "GePolCavity.hpp"
#include "PhysicalConstants.hpp"

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

class GePolCavityCsAddTest : public ::testing::Test
{
	protected:
		GePolCavity cavity;
		virtual void SetUp()
		{
  	                Eigen::Vector3d	C1( 0.0006122714,  0.0000000000,  0.0000000000);
  		        Eigen::Vector3d	H1( 1.5162556382, -1.3708721537,  0.0000000000);
  		        Eigen::Vector3d	H2(-0.7584339548,  0.6854360769,  1.7695110698);
  		        Eigen::Vector3d	H3(-0.7584339548,  0.6854360769, -1.7695110698);
			std::vector<Sphere> spheres;
			double radiusC = (1.70 * 1.20) / convertBohrToAngstrom;
			double radiusH = (1.20 * 1.20) / convertBohrToAngstrom;
			Sphere sph1(C1, radiusC);
			Sphere sph2(H1, radiusH);
			Sphere sph3(H2, radiusH);
			Sphere sph4(H3, radiusH);
			spheres.push_back(sph1);
			spheres.push_back(sph2);
			spheres.push_back(sph3);
			spheres.push_back(sph4);
			double area = 0.2 / convertBohr2ToAngstrom2;
			double probeRadius = 1.385 / convertBohrToAngstrom;
			// Addition of spheres is enabled, but will not happen in this particular case
			double minRadius = 0.2 / convertBohrToAngstrom;
			int pGroup = 1; // Cs 
			cavity = GePolCavity(spheres, area, probeRadius, minRadius, pGroup);
			fs::rename("PEDRA.OUT", "PEDRA.OUT.cs");
			fs::rename("cavity.off", "cavity.off.cs");
		}
};

TEST_F(GePolCavityCsAddTest, size)
{
	int size = 384;
	int actualSize = cavity.size();
	EXPECT_EQ(size, actualSize);
}

TEST_F(GePolCavityCsAddTest, irreducible_size)
{
	int size = 192;
	int actualSize = cavity.irreducible_size();
	EXPECT_EQ(size, actualSize);
}

TEST_F(GePolCavityCsAddTest, area)
{
	double area = 211.86178059383573; 
 	double actualArea = cavity.elementArea().sum();
	EXPECT_NEAR(area, actualArea, 1.0e-10);
}

TEST_F(GePolCavityCsAddTest, volume)
{
	double volume = 278.95706420724309;
	Eigen::Matrix3Xd elementCenter = cavity.elementCenter();
	Eigen::Matrix3Xd elementNormal = cavity.elementNormal();
	double actualVolume = 0;
        for ( int i = 0; i < cavity.size(); ++i )
	{
		actualVolume += cavity.elementArea(i) * elementCenter.col(i).dot(elementNormal.col(i));
	}
	actualVolume /= 3;
	EXPECT_NEAR(volume, actualVolume, 1.0e-10);
}
