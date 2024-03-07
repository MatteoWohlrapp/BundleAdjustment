
#include "ReconstructionError.h"

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

void ReconstructionError::zeroCenterPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    for (auto &point : *cloud)
    {
        point.x -= centroid[0];
        point.y -= centroid[1];
        point.z -= centroid[2];
    }
}

float ReconstructionError::computeUniformScaleFactor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2)
{

    float dimension1 = computePercentileDimension(cloud1, 0.0f);
    float dimension2 = computePercentileDimension(cloud2, 0.0f);

    // Uniform scale factor
    return dimension2 / dimension1;
}

void ReconstructionError::scalePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float scale)
{
    for (auto &point : *cloud)
    {
        point.x *= scale;
        point.y *= scale;
        point.z *= scale;
    }
}

double ReconstructionError::computeSelfReconstructionError(SceneMap *estimation)
{

    return computeReconstructionError(estimation, estimation);
}

double ReconstructionError::computeReconstructionError(SceneMap *estimation, std::string gtMeshPath)
{

    // Convert SceneMap to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloudEstimation(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloudTarget(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PLYReader Reader;
    if (Reader.read(gtMeshPath, *pclCloudTarget) == -1)
    {
        std::cout << "PLY file could not be read: " << gtMeshPath << std::endl;
    }

    for (const auto &point : estimation->getMapPoints())
    {
        Vector3f pos = point->getPosition();
        pclCloudEstimation->push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
    }
    /*
    pcl::PCLPointCloud2::Ptr estimationPCL2(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2::Ptr estimationFilteredPCL2(new pcl::PCLPointCloud2());

    pcl::toPCLPointCloud2(*pclCloudEstimation, *estimationPCL2);
    
    pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(transformedEstimationPCL2);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*transformedEstimationFilteredPCL2);
    */
    /*
    pcl::RadiusOutlierRemoval<pcl::PCLPointCloud2> outrem;
    // build the filter
    outrem.setInputCloud(estimationPCL2);
    outrem.setRadiusSearch(0.04);
    outrem.setMinNeighborsInRadius(2);
    // apply filter
    outrem.filter(*estimationFilteredPCL2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr estimationFiltered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromPCLPointCloud2(*estimationFilteredPCL2, *estimationFiltered);
    */

    //get gt pose of first camera, since our generated mesh is centered around this
    Eigen::Matrix4f pose = estimation->getKeyFrames()[0]->getGTPose().inverse();

    // Transform the world coordinate
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedEstimation(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*pclCloudEstimation, *transformedEstimation, pose);
 
    // Now proceed with zero-centering and scaling as before
    zeroCenterPointCloud(transformedEstimation);
    zeroCenterPointCloud(pclCloudTarget);

    float scaleFactor = computeUniformScaleFactor(transformedEstimation, pclCloudTarget);
    scalePointCloud(transformedEstimation, scaleFactor);

    pcl::io::savePLYFile((m_outputFilePath + std::string("_readin_room0_mesh.ply")), *pclCloudTarget);
    pcl::io::savePLYFile((m_outputFilePath + std::string("estimated_room0_mesh.ply")), *transformedEstimation);

    return computeReconstructionErrorPCL(transformedEstimation, pclCloudTarget);
}

double ReconstructionError::computeReconstructionError(SceneMap *estimation, SceneMap *target)
{

    // Convert SceneMap to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloudEstimation(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloudTarget(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto &point : estimation->getMapPoints())
    {
        Vector3f pos = point->getPosition();
        pclCloudEstimation->push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
    }

    for (const auto &point : target->getMapPoints())
    {
        Vector3f pos = point->getPosition();
        pclCloudTarget->push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
    }

    return computeReconstructionErrorPCL(pclCloudEstimation, pclCloudTarget);
}

double ReconstructionError::computeReconstructionErrorPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloudEstimation, pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloudTarget)
{

    // Proceed with ICP as before, using the transformed point clouds
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(pclCloudEstimation);
    icp.setInputTarget(pclCloudTarget);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    // Create a new point cloud with color information
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Add points from the target cloud in red
    for (const auto &point : *pclCloudTarget)
    {
        pcl::PointXYZRGB coloredPoint;
        coloredPoint.x = point.x;
        coloredPoint.y = point.y;
        coloredPoint.z = point.z;
        coloredPoint.r = 255; // Red
        coloredPoint.g = 0;
        coloredPoint.b = 0;
        coloredCloud->push_back(coloredPoint);
    }

    // Add points from the aligned estimation cloud in green
    for (const auto &point : Final)
    {
        pcl::PointXYZRGB coloredPoint;
        coloredPoint.x = point.x;
        coloredPoint.y = point.y;
        coloredPoint.z = point.z;
        coloredPoint.r = 0;
        coloredPoint.g = 255; // Green
        coloredPoint.b = 0;
        coloredCloud->push_back(coloredPoint);
    }

    // Save the combined cloud to a file
    pcl::io::savePLYFile((m_outputFilePath + std::string("_combined_colored_cloud.ply")), *coloredCloud);

    // Check if the alignment has converged and print the transformation
    if (!icp.hasConverged())
    {
        std::cerr << "ICP did not converge." << std::endl;
        return std::numeric_limits<float>::max(); // Return max error
    }

    // Calculate the difference or error after alignment
    double score = icp.getFitnessScore();

    std::cout << "ICP has converged, score is " << score << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    return score;
}

double ReconstructionError::computeErrorFromPLYFiles(const std::string &plyFile1, const std::string &plyFile2)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PLYReader Reader;
    if (Reader.read(plyFile1, *cloud1) == -1)
    {
        std::cerr << "PLY file could not be read: " << plyFile1 << std::endl;
        return std::numeric_limits<float>::max();
    }
    if (Reader.read(plyFile2, *cloud2) == -1)
    {
        std::cerr << "PLY file could not be read: " << plyFile2 << std::endl;
        return std::numeric_limits<float>::max();
    }

    return computeReconstructionErrorPCL(cloud1, cloud2);
}

float ReconstructionError::nthPercentile(std::vector<float> &data, float nth)
{
    if (data.empty())
        return 0.0f;

    size_t n = static_cast<size_t>((nth / 100.0) * (data.size()-1));
    std::nth_element(data.begin(), data.begin() + n, data.end());
    return data[n];
}

float ReconstructionError::computePercentileDimension(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float percentile)
{
    std::vector<float> xValues, yValues, zValues;
    for (const auto &point : *cloud)
    {
        xValues.push_back(point.x);
        yValues.push_back(point.y);
        zValues.push_back(point.z);
    }

    float x5 = nthPercentile(xValues, percentile);
    float x95 = nthPercentile(xValues, 100-percentile);
    float y5 = nthPercentile(yValues, percentile);
    float y95 = nthPercentile(yValues, 100-percentile);
    float z5 = nthPercentile(zValues, percentile);
    float z95 = nthPercentile(zValues, 100-percentile);

    float width = x95 - x5;
    float height = y95 - y5;
    float depth = z95 - z5;

    return std::max({width, height, depth});
}

Eigen::Matrix4f ReconstructionError::createSwapMatrix()
{
    Eigen::Matrix4f swapMatrix = Eigen::Matrix4f::Identity();

    // Swap the dimensions: x -> y, -y -> z, -z -> x
    swapMatrix(0, 0) = 0;
    swapMatrix(0, 2) = -1;
    swapMatrix(1, 0) = 1;
    swapMatrix(1, 1) = 0;
    swapMatrix(2, 1) = -1;
    swapMatrix(2, 2) = 0;

    return swapMatrix;
}