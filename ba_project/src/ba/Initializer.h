#pragma once

#include "Optimizer.h"
#include <Eigen/Dense>
#include "SfMHelper.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

enum InitType
{
	Standard,
	GTDepth
};

class Initializer
{
public:
	Initializer(std::shared_ptr<Frame>refFrame, InitType type, SceneMap *sceneMap, SfMHelper *sfmHelper);

	// initialize pose and scene map
	bool initialize(std::shared_ptr<Frame>currentFrame, std::vector<cv::DMatch> matches);

	// set refFrame
	void setRefFrame(std::shared_ptr<Frame>refFrame) { m_refFrame = refFrame; }

private:
	// initialize pose using ground truth depth
	bool initWithGTDepth(std::shared_ptr<Frame>currentFrame, std::vector<cv::DMatch> matches);

	// initialize pose using 2D-2D correspondences
	bool initPose(std::shared_ptr<Frame>currentFrame, std::vector<cv::DMatch> matches);

	// initialize scene map
	void initMap(std::shared_ptr<Frame>currentFrame, std::vector<cv::DMatch> matches);

	InitType m_type;
	std::shared_ptr<Frame>m_refFrame;
	SceneMap *m_sceneMap;
	SfMHelper *m_sfMHelper;
};