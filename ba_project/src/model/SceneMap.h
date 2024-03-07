#pragma once

#include <set>
#include "../utils/Eigen.h"
#include "MapPoint.h"
#include "Frame.h"
#include <thread>
#include <mutex>

using namespace std;

// holds all keyframes with poses and mappoints
class SceneMap
{
protected:
	struct cmp
	{
		bool operator()(std::shared_ptr<Frame>a, std::shared_ptr<Frame>b) const
		{
			return a->getID() < b->getID();
		}
	};

	std::set<std::shared_ptr<Frame>, cmp> m_keyframes;
	std::set<std::shared_ptr<MapPoint>> m_mapPoints;

public:
	SceneMap();
	// Getters and setters for scene map attributes
	void addKeyFrame(std::shared_ptr<Frame>frame);
	void addMapPoint(std::shared_ptr<MapPoint>point);
	void eraseMapPoint(std::shared_ptr<MapPoint>point);
	std::vector<std::shared_ptr<Frame>> getKeyFrames();
	std::vector<std::shared_ptr<MapPoint>> getMapPoints();
	std::vector<float> getMutableCameraForFrame(Frame &frame);
	std::vector<float> getMutable3DPointForMapPoint(MapPoint &mapPoint);
	int getMapPointCount();
	int getKeyFrameCount();
	void eraseKeyFrame(std::shared_ptr<Frame>frame);

private:
	std::mutex m_mapMutex;
	std::mutex m_frameMutex;
};