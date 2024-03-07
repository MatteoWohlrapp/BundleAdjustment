#include "Initializer.h"

using namespace cv;

Initializer::Initializer(std::shared_ptr<Frame>refFrame, InitType type, SceneMap *sceneMap, SfMHelper *sfmHelper)
	: m_refFrame(refFrame), m_type(type), m_sceneMap(sceneMap), m_sfMHelper(sfmHelper)
{
}

bool Initializer::initialize(std::shared_ptr<Frame>currentFrame, std::vector<DMatch> matches)
{
	bool success = false;

	// initialize pose
	switch (m_type)
	{
	case Standard:
		success = initPose(currentFrame, matches);
		break;
	case GTDepth:
		return initWithGTDepth(currentFrame, matches);
		break;
	default:
		break;
	}

	if (success)
	{
		// initialize map
		initMap(currentFrame, matches);
		return true;
	}
	return false;
}

bool Initializer::initWithGTDepth(std::shared_ptr<Frame>currentFrame, std::vector<DMatch> matches)
{
	// initial pose is identity
	// initalize dense map using detected features in refFrame and the corresponding GT depth values
	std::vector<KeyPoint> refKeypoints = m_refFrame->getKeypoints();
	cv::Mat refDepth = m_refFrame->getDepth();
	Matrix3f invIntr = m_refFrame->getIntrinsics().inverse();
	Matrix4f pose = m_refFrame->getPose(); // m_refFrame->getGTPose();

	for (int i = 0; i < refKeypoints.size(); i++)
	{
		KeyPoint p = refKeypoints[i];

		// get depth from depthMap
		// pixel coordinates are subpixel, so we have to extract subpixel value
		cv::Mat interpolated;
		cv::getRectSubPix(refDepth, cv::Size(1, 1), p.pt, interpolated);
		float depth = interpolated.at<float>(0, 0);

		// sort out points with invalid depth measurement
		if (depth == MINF)
		{
			continue;
		}

		// deproject point to camera coord
		Vector2f pixelCoords(p.pt.x, p.pt.y);
		Vector3f cameraSpace = (invIntr * pixelCoords.homogeneous()) * depth;

		// convert to world coordinates (could be omitted since initial pose is identity)
		Vector3f worldSpace = (pose * cameraSpace.homogeneous()).hnormalized();

		// Create Map point and associate with frame
		std::shared_ptr<MapPoint>mapPoint = std::make_shared<MapPoint>(worldSpace, m_refFrame, i);
		m_refFrame->addAssociatedMapPoint(i, mapPoint);

		// Add to SceneMap
		m_sceneMap->addMapPoint(mapPoint);
	}

	// perform pose estimation for source frame
	// initialize with identity and hope dense map helps to get a good pose estimate
	currentFrame->setPose(Matrix4f::Identity());

	std::vector<DMatch> noMapPointInRef;

	// add mappoint references based on matches with refframe
	for (DMatch m : matches)
	{
		int refIndex = m.queryIdx;
		int sourceIndex = m.trainIdx;

		std::shared_ptr<MapPoint>point = m_refFrame->getMapPoint(refIndex);

		if (point != nullptr)
		{
			// add cross reference
			point->addObservation(currentFrame, sourceIndex);
			currentFrame->addAssociatedMapPoint(sourceIndex, point);
		}
		else
		{
			// triangulate after pose refinement
			noMapPointInRef.push_back(m);
		}
	}
	//cout << "# matches: " << matches.size() << endl;
	//cout << "# correspondences: " << matches.size() - noMapPointInRef.size() << endl;

	//cout << "Initial Pose \n" << currentFrame->getPose() << endl;
	// optimize pose with motion only BA
	MotionOnlyBAOptimizerAngles optimizer = MotionOnlyBAOptimizerAngles();
	optimizer.setNbOfIterations(4);
	optimizer.setNbOfMaxItPerBA(20);
	optimizer.optimizeCameraPose(currentFrame);
	//cout << "Optimized Pose \n" << currentFrame->getPose() << endl;

	Matrix4f relPose = m_refFrame->getPose().inverse() * currentFrame->getPose();
	Matrix4f relGTPose = m_refFrame->getGTPose().inverse() * currentFrame->getGTPose();
	//cout << "Relative Pose \n" << relPose << endl;
	//cout << "Relative GT Pose \n" << relGTPose << endl;

	// triangulate remaining matches
	m_sfMHelper->triangulatePoints(m_refFrame, currentFrame, &noMapPointInRef, m_sceneMap, false);

	// add keyframes
	m_sceneMap->addKeyFrame(m_refFrame);
	m_sceneMap->addKeyFrame(currentFrame);

	return true;
}

bool Initializer::initPose(std::shared_ptr<Frame>currentFrame, std::vector<DMatch> matches)
{
	// Compute the relative pose between frame1 and frame2
	Matrix4f relativePose;

	if (m_sfMHelper->recoverPose(m_refFrame, currentFrame, &matches, relativePose))
	{

		//cout << "Initial Pose: \n" << relativePose << endl;
		// Set pose of second frame
		currentFrame->setPose(relativePose);

		/*
		// relative pose of currentframe in refframe coordinate system
		Matrix4f relPose = m_refFrame->getPose().inverse() * currentFrame->getPose();
		Matrix4f relGTPose = m_refFrame->getGTPose().inverse() * currentFrame->getGTPose();
		// cout << "Relative Pose \n" << relPose << endl;
		cout << "Relative GT Pose \n"
			 << relGTPose << endl;

		float dev = (relGTPose - relPose).block(0, 0, 3, 3).lpNorm<1>();
		cout << "Rotation Deviation: " << dev << endl;*/

		return true;
	}
	else
	{
		return false;
	}
}

void Initializer::initMap(std::shared_ptr<Frame>currentFrame, std::vector<DMatch> matches)
{
	// sort out outliers
	std::vector<DMatch> triangulateMatches;
	for (auto &match : matches)
	{
		if (!m_refFrame->isOutlier(match.queryIdx) && !currentFrame->isOutlier(match.trainIdx))
		{
			triangulateMatches.push_back(match);
		}
	}

	// Triangulate initial 3D points
	if (triangulateMatches.size() > 0)
	{
		m_sfMHelper->triangulatePoints(m_refFrame, currentFrame, &triangulateMatches, m_sceneMap, false);
	}

	// Add initial frames and points to the map
	m_sceneMap->addKeyFrame(m_refFrame);
	m_sceneMap->addKeyFrame(currentFrame);

	// Global BA
	GlobalBAOptimizerAngles optimizer = GlobalBAOptimizerAngles();
	optimizer.optimizeCamerasAndMapPoints(m_sceneMap, false,0);

	//cout << "Optimized Pose: \n" << currentFrame->getPose() << endl;
}
