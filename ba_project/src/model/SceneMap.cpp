#include "SceneMap.h"

SceneMap::SceneMap()
{
}

void SceneMap::addKeyFrame(std::shared_ptr<Frame>frame)
{
	frame->setKeyFrame();

	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_keyframes.insert(frame);
}

void SceneMap::addMapPoint(std::shared_ptr<MapPoint>point)
{
	std::lock_guard<std::mutex> lock(m_mapMutex);
	m_mapPoints.insert(point);
}

void SceneMap::eraseMapPoint(std::shared_ptr<MapPoint>point)
{
	std::lock_guard<std::mutex> lock(m_mapMutex);
	m_mapPoints.erase(point);
}

void SceneMap::eraseKeyFrame(std::shared_ptr<Frame>frame)
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	m_keyframes.erase(frame);
}

std::vector<std::shared_ptr<Frame>> SceneMap::getKeyFrames()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return vector<std::shared_ptr<Frame>>(m_keyframes.begin(), m_keyframes.end());
}

std::vector<std::shared_ptr<MapPoint>> SceneMap::getMapPoints()
{
	std::lock_guard<std::mutex> lock(m_mapMutex);
	return vector<std::shared_ptr<MapPoint>>(m_mapPoints.begin(), m_mapPoints.end());
}

std::vector<float> SceneMap::getMutableCameraForFrame(Frame &frame)
{
	return std::vector<float>();
}

std::vector<float> SceneMap::getMutable3DPointForMapPoint(MapPoint &mapPoint)
{
	return std::vector<float>();
}

int SceneMap::getMapPointCount()
{
	std::lock_guard<std::mutex> lock(m_mapMutex);
	return m_mapPoints.size();
}

int SceneMap::getKeyFrameCount()
{
	std::lock_guard<std::mutex> lock(m_frameMutex);
	return m_keyframes.size();
}
