#pragma once

#include <mutex>
#include <vector>
#include <iostream>
#include "../utils/Eigen.h"

#include "../ba/FeatureProcessor.h"
#include "MapPoint.h"
#include "opencv2/core.hpp"
#include <opencv2/core/eigen.hpp>
#include <memory.h>

using namespace std;

class Frame : public std::enable_shared_from_this<Frame>
{
public:
	Frame(cv::Mat color, cv::Mat depth, Matrix4f gtPose, Matrix3f intrinsics, FeatureProcessor *fp, int FrameID, const double timeStamp);

	void setPose(Matrix4f pose);
	void setColor(cv::Mat color);
	void setDepth(cv::Mat depth);
	void setDescriptors(cv::Mat descriptors);
	void setKeypoints(vector<cv::KeyPoint> keypoints);
	void setCorners(vector<cv::Point2f> corners);
	void setMapPoints(std::vector<std::shared_ptr<MapPoint>> mappoints);
	void addAssociatedMapPoint(int index, std::shared_ptr<MapPoint>mapPoint);
	void eraseAssociatedMapPoint(int index);
	void setOutlier(int index);
	void setInlier(int index);
	Matrix4f getPose();
	Matrix4f getGTPose();
	bool hasValidPose();
	std::vector<std::shared_ptr<MapPoint>> getMapPoints();
	std::shared_ptr<MapPoint>getMapPoint(int index);
	cv::Mat getColor();
	cv::Mat getDepth();
	Matrix3f getIntrinsics();
	cv::Mat getDescriptors();
	vector<cv::KeyPoint> getKeypoints();
	cv::KeyPoint *getKeypoint(int index);
	cv::Mat getDescriptor(int index);
	int getKeypointCount();
	vector<cv::Point2f> getCorners();
	bool isOutlier(int index);
	int getID();
	Vector3f getWorldPos();
	double getTimeStamp();
	int getHeight();
	int getWidth();

	void addCovisibilityFrame(std::shared_ptr<Frame>frame, const int &weight);
	void removeCovisibilityFrame(std::shared_ptr<Frame>frame);
	void updateCovisibilityGraph();
	std::vector<std::shared_ptr<Frame>> getBestCovisibilityFrames(int n);
	std::vector<std::shared_ptr<Frame>> getAllCovisibilityFrames();
	void init(); // initialize frame

	float getMedianMapPointDepth();

	void erase();
	void setKeyFrame();
	bool isKeyFrame();

	template <typename T>
	std::ostream &operator<<(std::ostream &outs)
	{
		return outs << "()";
	}

private:
	int m_frameID;

	// gt attributes
	cv::Mat m_color;
	cv::Mat m_depth;
	Matrix3f m_intrinsics;
	Matrix4f m_gtPose;

	cv::Mat m_descriptors;
	// indexes from keypoint and mapPoints should be connected
	vector<cv::KeyPoint> m_keypoints;
	vector<std::shared_ptr<MapPoint>> m_mapPoints;
	std::vector<bool> m_outlier;

	Matrix4f m_pose;
	bool m_validPose = false;

	std::map<std::shared_ptr<Frame>, int> m_covisibilityWeights;
	std::vector<std::shared_ptr<Frame>> m_orderedCovisibilityFrames;
	// std::vector<int> m_orderedCovisibilityWeights;

	// for corner detectors
	vector<cv::Point2f> m_corners;
	FeatureProcessor *m_fp;

	std::mutex m_frameMutex;

	double m_timeStamp;
	bool isCulled = false;
	bool m_isKeyFrame = false;
};
