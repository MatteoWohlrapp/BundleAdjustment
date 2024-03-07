#include "Frame.h"
#include <opencv2/core/eigen.hpp>
Frame::Frame(cv::Mat color, cv::Mat depth, Matrix4f gtPose, Matrix3f intrinsics, FeatureProcessor *fp, int frameID, double timeStamp)
	: m_pose(Matrix4f::Identity()), m_depth(depth), m_color(color), m_gtPose(gtPose), m_intrinsics(intrinsics), m_fp(fp), m_frameID(frameID), m_timeStamp(timeStamp)
{
}

void Frame::init()
{
	if (m_fp)
	{
		m_fp->detectFeatures(shared_from_this());
	}
	m_mapPoints = std::vector<std::shared_ptr<MapPoint>>(m_keypoints.size(), nullptr);
	m_outlier = std::vector<bool>(m_keypoints.size(), false);
}

void Frame::setPose(Matrix4f pose)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_pose = pose;
	m_validPose = true;
}

void Frame::setColor(cv::Mat color)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_color = color;
}

void Frame::setDepth(cv::Mat depth)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_depth = depth;
}

void Frame::setDescriptors(cv::Mat descriptors)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_descriptors = descriptors;
}

void Frame::setKeypoints(vector<cv::KeyPoint> keypoints)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_keypoints = keypoints;
	m_mapPoints = std::vector<std::shared_ptr<MapPoint>>(m_keypoints.size(), nullptr);
	m_outlier = std::vector<bool>(m_keypoints.size(), false);
}

void Frame::setCorners(vector<cv::Point2f> corners)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_corners = corners;
}

void Frame::setMapPoints(vector<std::shared_ptr<MapPoint>> mappoints)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_mapPoints = mappoints;
}

void Frame::addAssociatedMapPoint(int index, std::shared_ptr<MapPoint>mapPoint)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_mapPoints[index] = mapPoint;
}

void Frame::eraseAssociatedMapPoint(int index)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_mapPoints[index] = nullptr;
}

void Frame::setOutlier(int index)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_outlier[index] = true;
}

void Frame::setInlier(int index)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_outlier[index] = false;
}

Matrix4f Frame::getPose()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_pose;
}

Matrix4f Frame::getGTPose()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_gtPose;
}

bool Frame::hasValidPose()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_validPose;
}

std::vector<std::shared_ptr<MapPoint>> Frame::getMapPoints()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_mapPoints;
}

std::shared_ptr<MapPoint>Frame::getMapPoint(int index)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	if (index >= 0 && index < m_mapPoints.size())
	{
		return m_mapPoints[index];
	}
	return nullptr;
}

cv::Mat Frame::getColor()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_color;
}

cv::Mat Frame::getDepth()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_depth;
}

Matrix3f Frame::getIntrinsics()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_intrinsics;
}

cv::Mat Frame::getDescriptors()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_descriptors;
}

vector<cv::KeyPoint> Frame::getKeypoints()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_keypoints;
}

cv::KeyPoint *Frame::getKeypoint(int index)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	if (index >= m_keypoints.size())
	{
		return nullptr;
	}
	return &m_keypoints[index];
}

cv::Mat Frame::getDescriptor(int index)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_descriptors.row(index);
}

int Frame::getKeypointCount()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_keypoints.size();
}

vector<cv::Point2f> Frame::getCorners()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_corners;
}

bool Frame::isOutlier(int index)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_outlier[index];
}

int Frame::getID()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_frameID;
}

Vector3f Frame::getWorldPos()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_pose.block(0, 3, 3, 1);
}

double Frame::getTimeStamp()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_timeStamp;
}

int Frame::getHeight()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_color.rows;
}

int Frame::getWidth()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_color.cols;
}

void Frame::addCovisibilityFrame(std::shared_ptr<Frame>frame, const int &weight)
{
	std::map<std::shared_ptr<Frame>, int> covisibilityWeights;
	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		covisibilityWeights = m_covisibilityWeights;
	}

	// insert into map, if already existing, only update weight
	if (covisibilityWeights.find(frame) != covisibilityWeights.end())
	{
		covisibilityWeights[frame] = weight;
	}
	else
	{
		covisibilityWeights.insert(std::make_pair(frame, weight));
	}

	// update best covisible frames
	// flip map, so we can sort after weight
	vector<pair<int, std::shared_ptr<Frame>>> inv;
	for (auto it = covisibilityWeights.begin(); it != covisibilityWeights.end(); it++)
	{
		inv.push_back(make_pair(it->second, it->first));
	}

	std::sort(inv.begin(), inv.end());

	std::vector<std::shared_ptr<Frame>> sortedFrames;
	for (int i = 0; i < inv.size(); i++)
	{
		sortedFrames.push_back(inv[i].second);
	}

	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		m_orderedCovisibilityFrames = vector<std::shared_ptr<Frame>>(sortedFrames.begin(), sortedFrames.end());
		m_covisibilityWeights = covisibilityWeights;
	}
}

void Frame::removeCovisibilityFrame(std::shared_ptr<Frame>frame)
{
	std::map<std::shared_ptr<Frame>, int> covisibilityWeights;
	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		covisibilityWeights = m_covisibilityWeights;
	}

	if (covisibilityWeights.find(frame) != covisibilityWeights.end())
	{
		covisibilityWeights.erase(frame);

		// update best covisible frames
		// flip map, so we can sort after weight
		vector<pair<int, std::shared_ptr<Frame>>> inv;
		for (auto it = covisibilityWeights.begin(); it != covisibilityWeights.end(); it++)
		{
			inv.push_back(make_pair(it->second, it->first));
		}

		std::sort(inv.begin(), inv.end());

		std::vector<std::shared_ptr<Frame>> sortedFrames;
		for (int i = 0; i < inv.size(); i++)
		{
			sortedFrames.push_back(inv[i].second);
		}

		{
			std::lock_guard<std::mutex> lock(m_frameMutex);
			m_orderedCovisibilityFrames = vector<std::shared_ptr<Frame>>(sortedFrames.begin(), sortedFrames.end());
			m_covisibilityWeights = covisibilityWeights;
		}
	}
}

void Frame::updateCovisibilityGraph()
{
	vector<std::shared_ptr<MapPoint>> mapPoints;
	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		mapPoints = m_mapPoints;
	}
	// create covisibility graph
	std::map<std::shared_ptr<Frame>, int> covisibilityWeights;
	for (int i = 0; i < mapPoints.size(); i++)
	{
		std::shared_ptr<MapPoint>mp = mapPoints[i];

		if (mp && !isOutlier(i))
		{
			map<std::shared_ptr<Frame>, size_t> observations = mp->getObservingKeyframes();

			for (auto it = observations.begin(); it != observations.end(); it++)
			{

				if (it->first == shared_from_this())
				{
					continue;
				}
				else
				{
					covisibilityWeights[it->first]++;
				}
			}
		}
	}

	// add edges to other frames if #connections>threshold
	int maxConnections = 0;
	std::shared_ptr<Frame>maxFrame = NULL;
	int th = 10;

	// for sorting
	vector<pair<int, std::shared_ptr<Frame>>> inv;
	std::map<std::shared_ptr<Frame>, int> filteredCovisibilityWeights;
	for (auto it = covisibilityWeights.begin(); it != covisibilityWeights.end(); it++)
	{
		if (it->second > maxConnections)
		{
			maxFrame = it->first;
			maxConnections = it->second;
		}

		if (it->second > th && it->first->isKeyFrame())
		{
			inv.push_back(make_pair(it->second, it->first));

			// add edge
			it->first->addCovisibilityFrame(shared_from_this(), it->second);

			filteredCovisibilityWeights.insert(pair<std::shared_ptr<Frame>, int>(it->first, it->second));
		}
	}

	// in case no frame has more than threshold connections, we at least add frame with maximal connections
	if (inv.empty())
	{
		inv.push_back(make_pair(maxConnections, maxFrame));
		maxFrame->addCovisibilityFrame(shared_from_this(), maxConnections);
		filteredCovisibilityWeights.insert(pair<std::shared_ptr<Frame>, int>(maxFrame, maxConnections));
	}

	// compute sorted list for best covisibility frames
	sort(inv.begin(), inv.end());

	std::vector<std::shared_ptr<Frame>> sortedFrames;
	for (int i = 0; i < inv.size(); i++)
	{
		sortedFrames.push_back(inv[i].second);
	}

	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		// update current graph with new one
		m_covisibilityWeights = filteredCovisibilityWeights;
		m_orderedCovisibilityFrames = vector<std::shared_ptr<Frame>>(sortedFrames.begin(), sortedFrames.end());
	}
}

std::vector<std::shared_ptr<Frame>> Frame::getBestCovisibilityFrames(int n)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	// return best covisibilities
	if (m_orderedCovisibilityFrames.size() <= n)
	{
		return m_orderedCovisibilityFrames;
	}

	return std::vector<std::shared_ptr<Frame>>(m_orderedCovisibilityFrames.begin(), m_orderedCovisibilityFrames.begin() + n);
}

std::vector<std::shared_ptr<Frame>> Frame::getAllCovisibilityFrames()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_orderedCovisibilityFrames;
}

float Frame::getMedianMapPointDepth()
{
	std::vector<std::shared_ptr<MapPoint>> mapPoints;
	Matrix4f extr;
	{
		std::lock_guard<std::mutex> lock(m_frameMutex);
		extr = m_pose.inverse();
		mapPoints = m_mapPoints;
	}

	std::vector<float> depths;
	depths.reserve(mapPoints.size());
	for (int i = 0; i < mapPoints.size(); i++)
	{
		if (mapPoints[i])
		{
			depths.push_back((extr * mapPoints[i]->getPosition().homogeneous()).hnormalized().z());
		}
	}

	std::sort(depths.begin(), depths.end());

	return depths[(depths.size() - 1) / 2];
}

void Frame::erase()
{
	// erase covisibility graph edges
	for (auto it = m_covisibilityWeights.begin(); it != m_covisibilityWeights.end(); it++)
	{
		it->first->removeCovisibilityFrame(shared_from_this());
	}

	// erase observations
	for (int i = 0; i < m_mapPoints.size(); i++)
	{
		if (m_mapPoints[i])
		{
			m_mapPoints[i]->removeObservation(shared_from_this());
		}
	}

	std::lock_guard<std::mutex> lock(m_frameMutex);
	isCulled = true;
	m_covisibilityWeights.clear();
	m_mapPoints.clear();
}

void Frame::setKeyFrame()
{
	m_isKeyFrame = true;
}

bool Frame::isKeyFrame()
{
	return m_isKeyFrame;
}
