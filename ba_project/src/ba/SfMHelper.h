#pragma once

#include <Eigen/Geometry>
#include <vector>
#include <map>
#include "Optimizer.h"
#include "../utils/Eigen.h"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

enum EstimationType
{
    PNP,
    BA,
    EssentialOrHomography
};

class SfMHelper
{
public:

    static SfMHelper* getInstance();

    SfMHelper(SfMHelper const&) = delete;
    void operator=(SfMHelper const&) = delete;

    void setPoseEstimationType(EstimationType type);

    // Function to estimate the pose of the current frame, given the estimation type
    void estimatePose(std::shared_ptr<Frame>frame, std::shared_ptr<Frame> lastFrame, std::vector<cv::DMatch>* matches);

    // Estimate a pose based on 2D-2D correspondences
    bool recoverPose(std::shared_ptr<Frame>frame1, std::shared_ptr<Frame>frame2, std::vector<cv::DMatch> *matches, Matrix4f &pose);

    Eigen::Matrix4f convertToMatrix4f(cv::Mat R, cv::Mat t);

    // Triangulate points from two frames to create 3D points
    void triangulatePoints(std::shared_ptr<Frame>frame1, std::shared_ptr<Frame>frame2, std::vector<cv::DMatch> *matches, SceneMap *map, bool checkBaseline);

    cv::Mat getCVProjectionMatrix(std::shared_ptr<Frame>frame);

    Eigen::Matrix4f setUpDebugCam(std::vector<cv::Point3f> p3d, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, cv::Mat &intr);
    
    void searchInNeighbors(std::shared_ptr<Frame>frame, FeatureProcessor *fp, SceneMap *map);

    // Function to get initial pose from constant speed assumption
    Eigen::Matrix4f getPoseFromConstantSpeed(SceneMap *map);

    void cullRecentMapPoints(std::shared_ptr<Frame>currentFrame, SceneMap *map);

    void cullRedundantKeyframes(std::shared_ptr<Frame>currentFrame, SceneMap *map);

    void eraseOutlier(SceneMap *map, int currentFrameID);

private:
    static SfMHelper* inst_;   // The one, single instance

    SfMHelper() : estimation(EstimationType::BA){};

    //~SfMHelper();

    EstimationType estimation;
    std::vector<std::shared_ptr<MapPoint>> m_recentlyAddedPoints;

    void estimatePoseUsingPnP(std::shared_ptr<Frame> frame);
    void estimatePoseUsingBA(std::shared_ptr<Frame> frame);
};
