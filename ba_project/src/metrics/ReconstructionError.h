#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include "../model/SceneMap.h"

// class to compute reconstruction error between two point clouds
class ReconstructionError
{

public:
    // constructor
    ReconstructionError(std::string outputFilePath): m_outputFilePath(outputFilePath) {};

    // destructor
    ~ReconstructionError(){};

    double computeReconstructionErrorPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloudEstimation, pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloudTarget);

    // compute reconstruction error between two point clouds
    double computeReconstructionError(SceneMap *estimation, SceneMap *target);

    // compute reconstruction error between estimated point cloud and ground truth mesh file
    double computeReconstructionError(SceneMap *estimation, std::string gtMeshPath);

    double computeSelfReconstructionError(SceneMap *estimation);

    double computeErrorFromPLYFiles(const std::string &plyFile1, const std::string &plyFile2);

private:
    void zeroCenterPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    void scalePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float scale);
    float computeUniformScaleFactor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2);
    float computePercentileDimension(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float percentile);
    float nthPercentile(std::vector<float>& data, float nth);
    Eigen::Matrix4f createSwapMatrix();
    std::string m_outputFilePath;
};