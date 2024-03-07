#pragma once

#include <mutex>
#include <vector>
#include "../utils/Eigen.h"
#include <opencv2/core.hpp>
#include <memory.h>

// remove circular dependency
class Frame;

using namespace std;

class MapPoint : public std::enable_shared_from_this<MapPoint>
{
public:
	MapPoint(Vector3f worldPosition, std::shared_ptr<Frame>observingFrame, size_t indexInFrame);

	Vector2f getCorresponding2DKeyPointPosition(std::shared_ptr<Frame>frame);

	Vector3f getPosition();
	void setPosition(Vector3f position);

	void addObservation(std::shared_ptr<Frame>observingFrame, size_t index);

	void addObservationFast(std::shared_ptr<Frame>observingFrame, size_t index);

	void replaceObservation(std::shared_ptr<Frame>observingFrame, size_t index);

	void removeObservation(std::shared_ptr<Frame>frame);

	Vector3f getViewingDirection();

	cv::Mat getDescriptor();

	std::map<std::shared_ptr<Frame>, size_t> getObservingKeyframes();

	int getNumObserved();

	bool fuse(std::shared_ptr<MapPoint>other);

	void computeViewingDirection();

	void computeDescriptor();

	bool isObservedBy(std::shared_ptr<Frame>frame);

	int getObservedIndex(std::shared_ptr<Frame>frame);

	std::shared_ptr<Frame>getReferenceFrame();

	Vector4uc getReferenceColor();

	float getMinDistance();
	float getMaxDistance();

	void erase();

	bool isInvalid();

	std::string toString()
	{
		return std::string("[" + std::to_string(m_worldPosition.x()) + ", " + std::to_string(m_worldPosition.y()) + ", " + std::to_string(m_worldPosition.z()) + "]");
	}

private:
	void computeReferenceColor();

	Vector3f m_worldPosition;
	Vector3f m_viewingDirection;
	std::map<std::shared_ptr<Frame>, size_t> m_observingKeyframes; // the (value, size_t) should be the index in the mapPoint list for the (key as) keyFrame
	Vector4uc m_referenceColor;

	cv::Mat m_descriptor;
	std::shared_ptr<Frame>m_referenceKeyframe;
	int m_referenceIndex;
	std::mutex m_pointMutex;

	float m_maxDist;
	float m_minDist;

	bool m_invalid = false;
};
