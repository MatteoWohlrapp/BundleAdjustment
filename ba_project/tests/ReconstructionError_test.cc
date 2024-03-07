#include "gtest/gtest.h"
#include "../src/ba/SfMHelper.h"
#include "../src/model/Frame.h"
#include "../src/model/SceneMap.h"
#include "../src/utils/Eigen.h"
#include "../src/ba/FeatureProcessor.h"
#include "opencv2/core.hpp"
#include <opencv2/core/eigen.hpp>
#include "../src/metrics/ReconstructionError.h"

class ReconstructionErrorTest : public ::testing::Test
{
protected:
    ReconstructionError* reconstructionError;
    void SetUp() override
    {    
        reconstructionError = new ReconstructionError();
    }

    void TearDown() override
    {
    }

    SceneMap* CreateMockPointCloud1() {
        SceneMap *sceneMap = new SceneMap();

        sceneMap->addMapPoint(std::make_shared<MapPoint>(Vector3f(0.0, 0.0, 0.0), nullptr, 0)); 
        sceneMap->addMapPoint(std::make_shared<MapPoint>(Vector3f(1.0, 0.0, 0.0), nullptr, 1));
        sceneMap->addMapPoint(std::make_shared<MapPoint>(Vector3f(0.0, 1.0, 0.0), nullptr, 2));

        return sceneMap;
    }

    SceneMap* CreateMockPointCloud2() {
        SceneMap *sceneMap = new SceneMap();

        sceneMap->addMapPoint(std::make_shared<MapPoint>(Vector3f(0.1, 0.1, 0.0), nullptr, 0)); 
        sceneMap->addMapPoint(std::make_shared<MapPoint>(Vector3f(1.1, 0.1, 0.0), nullptr, 1));
        sceneMap->addMapPoint(std::make_shared<MapPoint>(Vector3f(0.1, 1.1, 0.0), nullptr, 2));

        return sceneMap;
    }
};

TEST_F(ReconstructionErrorTest, CompareIdenticalClouds) {
        std::cout << " Before first point cloud created" << std::endl; 

    auto sceneMap1 = CreateMockPointCloud1();
    std::cout << "First point cloud created" << std::endl; 
    auto sceneMap2 = CreateMockPointCloud1(); // Using the same mock point cloud as cloud1

    double error = reconstructionError->computeReconstructionError(sceneMap1, sceneMap2);
    EXPECT_NEAR(error, 0.0, 1e-5); // The error should be near zero for identical clouds
}

TEST_F(ReconstructionErrorTest, CompareSimilarClouds) {
auto sceneMap1 = CreateMockPointCloud1();
    auto sceneMap2 = CreateMockPointCloud2(); // Using the same mock point cloud as cloud1

    double error = reconstructionError->computeReconstructionError(sceneMap1, sceneMap2);
    EXPECT_GT(error, 0.0); // The error should be greater than zero for different clouds
    EXPECT_LT(error, 0.1); // Assuming the error will be small for similar clouds
}