#include "FeatureProcessor.h"
#include "../model/Frame.h"

using namespace std;
using namespace cv;

bool FeatureProcessor::init(FeatureType type, bool drawMatches, bool drawFeatures)
{
	m_featureType = type;
	m_drawFeatures = drawFeatures;
	m_drawMatches = drawMatches;
	if (m_drawFeatures)
		cv::namedWindow("Features", cv::WindowFlags::WINDOW_AUTOSIZE);
	if (m_drawMatches)
		cv::namedWindow("Matches", cv::WindowFlags::WINDOW_NORMAL);
	if (type == SURF)
	{
		// init surf detector
		int minHessian = 300;
		m_surfDetector = cv::xfeatures2d::SURF::create(minHessian, 8);
	}

	return true;
}

void FeatureProcessor::preprocessFrame(std::shared_ptr<Frame>frame)
{
	// convert to RGB
	cv::Mat cvt;
	cvtColor(frame->getColor(), cvt, cv::COLOR_BGRA2RGB);

	// subtract mean intenstity
	//cv::Scalar meanIntensity = cv::mean(cvt);
	// meanIntensity[3] = 0; //discard alpha
	// - meanIntensity
	frame->setColor(cvt);
}

vector<cv::DMatch> FeatureProcessor::matchFeatures(std::shared_ptr<Frame>frame1, std::shared_ptr<Frame>frame2, float ratioThresh)
{
	// We don't have descriptors for corner methods yet, so skip corners for now
	if (m_featureType == HarrisCorner || m_featureType == ShiTomasi)
	{
		return vector<cv::DMatch>();
	}
	else
	{
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
		std::vector<std::vector<cv::DMatch>> knn_matches;
		matcher->knnMatch(frame1->getDescriptors(), frame2->getDescriptors(), knn_matches, 2);
		//-- Filter matches using the Lowe's ratio test
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratioThresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}

		// cross-check matches (FLANN matcher will return multiple matches per train point, we manually have to filter for this if we want unique matches)
		std::sort(good_matches.begin(), good_matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2)
				  { return m1.trainIdx < m2.trainIdx; });

		vector<cv::DMatch> unique_matches;
		for (int i = 0; i < good_matches.size(); i++)
		{
			if (i == 0)
			{
				unique_matches.push_back(good_matches[0]);
			}
			else if (good_matches[i].trainIdx != good_matches[i - 1].trainIdx)
			{
				unique_matches.push_back(good_matches[i]);
			}
		}

		//-- Draw matches
		if (m_drawMatches)
		{
			Mat img_matches;
			drawMatches(frame1->getColor(), frame1->getKeypoints(), frame2->getColor(), frame2->getKeypoints(), unique_matches, img_matches, Scalar::all(-1),
						Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			//-- Show detected matches
			int windowWidth = frame1->getColor().cols;
			int windowHeight = frame1->getColor().rows / 2;
			cv::resizeWindow("Matches", windowWidth, windowHeight);
			imshow("Matches", img_matches);
			waitKey();
		}
		// cout << "#Matches: " << unique_matches.size() << endl;
		return unique_matches;
	}
}

void FeatureProcessor::detectFeatures(std::shared_ptr<Frame>frame)
{

	preprocessFrame(frame);

	switch (m_featureType)
	{
	case FeatureType::SURF:
		surf(frame);
		break;
	case ShiTomasi:
		shiTomasi(frame);
		break;
	case HarrisCorner:
		harrisCorner(frame);
		break;
	default:
		break;
	}
}

void FeatureProcessor::harrisCorner(std::shared_ptr<Frame>frame)
{
	int thresh = 240;

	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	cv::Mat gray;
	cv::cvtColor(frame->getColor(), gray, cv::COLOR_BGR2GRAY);

	cv::Mat dst = cv::Mat::zeros(frame->getColor().size(), CV_32FC1);
	cornerHarris(gray, dst, blockSize, apertureSize, k);

	cv::Mat dst_norm, dst_norm_scaled;
	normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	vector<cv::Point2f> corners;
	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > thresh)
			{
				circle(dst_norm_scaled, cv::Point(j, i), 5, cv::Scalar(0), 2, 8, 0);
				corners.push_back(cv::Point2f(j, i));
			}
		}
	}

	frame->setCorners(corners);

	if (m_drawFeatures)
	{
		cv::imshow("Features", dst_norm_scaled);
		cv::waitKey();
	}
}

void FeatureProcessor::shiTomasi(std::shared_ptr<Frame>frame)
{
	int maxCorners = 10;

	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3, gradientSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;

	cv::Mat src_gray;
	cvtColor(frame->getColor(), src_gray, cv::COLOR_BGRA2GRAY);

	vector<cv::Point2f> corners;

	goodFeaturesToTrack(src_gray,
						corners,
						maxCorners,
						qualityLevel,
						minDistance,
						cv::Mat(),
						blockSize,
						gradientSize,
						useHarrisDetector,
						k);
	std::cout << "** Number of corners detected: " << frame->getCorners().size() << std::endl;

	if (m_drawFeatures)
	{
		cv::Mat copy = frame->getColor().clone();

		int radius = 4;
		cv::RNG rng(12345);
		for (size_t i = 0; i < corners.size(); i++)
		{
			circle(copy, corners[i], radius, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED);
		}

		imshow("Features", copy);
		cv::waitKey();
	}

	cv::Size winSize = cv::Size(5, 5);
	cv::Size zeroZone = cv::Size(-1, -1);
	cv::TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001);
	cornerSubPix(src_gray, corners, winSize, zeroZone, criteria);

	frame->setCorners(corners);

	for (size_t i = 0; i < corners.size(); i++)
	{
		std::cout << " -- Refined Corner [" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << std::endl;
	}
}

void FeatureProcessor::surf(std::shared_ptr<Frame>frame)
{

	//-- Step 1: Detect the keypoints using SURF Detector and create descriptors
	if (m_surfDetector)
	{
		vector<cv::KeyPoint> keypoints;
		Mat descriptors;

		m_surfDetector->detectAndCompute(frame->getColor(), noArray(), keypoints, descriptors);

		frame->setDescriptors(descriptors);
		frame->setKeypoints(keypoints);

		if (m_drawFeatures)
		{
			//-- Draw keypoints
			Mat img_keypoints;
			drawKeypoints(frame->getColor(), frame->getKeypoints(), img_keypoints);
			//-- Show detected (drawn) keypoints
			imshow("Features", img_keypoints);
			cv::waitKey();
		}
	}
	else
	{
		std::cout << "SURF detector not initialized!" << std::endl;
	}
}
