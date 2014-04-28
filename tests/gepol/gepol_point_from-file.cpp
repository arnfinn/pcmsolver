#include <vector>
#include <cmath>

#include "Config.hpp"

#include "EigenPimpl.hpp"

#include "GePolCavity.hpp"

#include "gtestPimpl.hpp"

class GePolCavityRestartTest : public ::testing::Test
{
protected:
    GePolCavity cavity;
    virtual void SetUp() {
        cavity.loadCavity("point.npz");
    }
};

/*! \class GePolCavity 
 *  \test \b GePolCavityRestartTest_size tests GePol cavity size for a point charge loading the cavity from a .npz file
 */
TEST_F(GePolCavityRestartTest, size)
{
    int size = 32;
    int actualSize = cavity.size();
    EXPECT_EQ(size, actualSize);
}

/*! \class GePolCavity 
 *  \test \b GePolCavityRestartTest_area tests GePol cavity surface area for a point charge loading the cavity from from a .npz file
 */
TEST_F(GePolCavityRestartTest, area)
{
    double area = 4.0 * M_PI * pow(1.0, 2);
    double actualArea = cavity.elementArea().sum();
    EXPECT_DOUBLE_EQ(area, actualArea);
//	EXPECT_NEAR(area, actualArea, 1.0e-12);
}

/*! \class GePolCavity 
 *  \test \b GePolCavityRestartTest_volume tests GePol cavity volume for a point charge loading the cavity from from a .npz file
 */
TEST_F(GePolCavityRestartTest, volume)
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
