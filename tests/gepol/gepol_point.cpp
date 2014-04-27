#include <vector>
#include <cmath>

#include "Config.hpp"

#include "EigenPimpl.hpp"

#include "GePolCavity.hpp"
#include "Symmetry.hpp"

#include "gtestPimpl.hpp"

class GePolCavityTest : public ::testing::Test
{
protected:
    GePolCavity cavity;
    virtual void SetUp() {
        Eigen::Vector3d origin(0.0, 0.0, 0.0);
        std::vector<Sphere> spheres;
        Sphere sph1(origin,  1.0);
        spheres.push_back(sph1);
        double area = 0.4;
        // C1
        Symmetry pGroup = buildGroup(0, 0, 0, 0);
        cavity = GePolCavity(spheres, area, 0.0, 100.0, pGroup);
        cavity.saveCavity("point.npz");
    }
};

TEST_F(GePolCavityTest, size)
{
    int size = 32;
    int actualSize = cavity.size();
    EXPECT_EQ(size, actualSize);
}

TEST_F(GePolCavityTest, area)
{
    double area = 4.0 * M_PI * pow(1.0, 2);
    double actualArea = cavity.elementArea().sum();
    EXPECT_DOUBLE_EQ(area, actualArea);
//	EXPECT_NEAR(area, actualArea, 1.0e-12);
}

TEST_F(GePolCavityTest, volume)
{
    double volume = 4.0 * M_PI * pow(1.0, 3) / 3.0;
    Eigen::Matrix3Xd elementCenter = cavity.elementCenter();
    Eigen::Matrix3Xd elementNormal = cavity.elementNormal();
    double actualVolume = 0;
    for ( int i = 0; i < cavity.size(); ++i ) {
        actualVolume += cavity.elementArea(i) * elementCenter.col(i).dot(elementNormal.col(
                            i));
    }
    actualVolume /= 3;
    EXPECT_DOUBLE_EQ(volume, actualVolume);
//	EXPECT_NEAR(volume, actualVolume, 1.0e-12);
}
