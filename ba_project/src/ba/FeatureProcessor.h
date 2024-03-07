#pragma once

#include <vector>
#include <iostream>

#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>

class Frame;

enum FeatureType
{
	HarrisCorner,
	ShiTomasi,
	SURF
};

using namespace std;

// Class to process features in the frame and match them
class FeatureProcessor
{

public:
	FeatureProcessor() {}

	~FeatureProcessor()
	{
		m_surfDetector.reset();
	}

	// Initialize the feature processor
	bool init(FeatureType type, bool drawMatches, bool drawFeatures = false);

	// Preprocess the frame
	void preprocessFrame(std::shared_ptr<Frame>frame);

	// Detect features between frames and return the matches
	vector<cv::DMatch> matchFeatures(std::shared_ptr<Frame>frame1, std::shared_ptr<Frame>frame2, float ratioThresh = 0.7f);

	// Detect features in the frame
	void detectFeatures(std::shared_ptr<Frame>frame);

	void harrisCorner(std::shared_ptr<Frame>frame);

	void shiTomasi(std::shared_ptr<Frame>frame);

	void surf(std::shared_ptr<Frame>frame);

private:
	FeatureType m_featureType = SURF;
	bool m_drawFeatures = false;
	bool m_drawMatches = false;
	cv::Ptr<cv::xfeatures2d::SURF> m_surfDetector;
};

#endif