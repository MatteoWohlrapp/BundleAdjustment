#include "MapPoint.h"
#include "Frame.h"
#include <memory.h>

using namespace cv;

MapPoint::MapPoint(Vector3f worldPosition, std::shared_ptr<Frame>observingFrame, size_t indexInFrame)
	: m_referenceKeyframe(observingFrame), m_worldPosition(worldPosition), m_referenceIndex(indexInFrame)
{
	m_observingKeyframes.insert(std::make_pair(observingFrame, indexInFrame));
	if (observingFrame)
	{
		m_descriptor = observingFrame->getDescriptors().row(indexInFrame).clone(); // observingFrame->getDescriptors()(Range(0, 1), Range::all()).clone();
		m_viewingDirection = (m_worldPosition - observingFrame->getPose().block(0, 3, 3, 1)).normalized();

		// set initial min and max depth
		float worldDist = (m_worldPosition - observingFrame->getWorldPos()).norm();
		float levelScale = pow(1.2, observingFrame->getKeypoint(indexInFrame)->octave);
		float maxScale = pow(1.2, 7);

		m_maxDist = worldDist * levelScale;
		m_minDist = m_maxDist / maxScale;

		// compute reference color;
		computeReferenceColor();
	}
}

Vector2f MapPoint::getCorresponding2DKeyPointPosition(std::shared_ptr<Frame>frame)
{

	std::lock_guard<std::mutex> lock(m_pointMutex);
	size_t mapPointIndexInFrame = m_observingKeyframes.find(frame)->second;
	Point2f point = frame->getKeypoint(mapPointIndexInFrame)->pt;

	return Vector2f(point.x, point.y);
}

Vector3f MapPoint::getPosition()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_worldPosition;
}

void MapPoint::setPosition(Vector3f position)
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	m_worldPosition = position;
}

void MapPoint::addObservation(std::shared_ptr<Frame>observingFrame, size_t index)
{
	{
		std::lock_guard<std::mutex> lock(m_pointMutex);
		m_observingKeyframes.insert(std::make_pair(observingFrame, index));
	}

	// select best descriptor of all observing frames and update mean viewing direction
	computeDescriptor();
	computeViewingDirection();
}

void MapPoint::addObservationFast(std::shared_ptr<Frame>observingFrame, size_t index)
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	m_observingKeyframes.insert(std::make_pair(observingFrame, index));
}

void MapPoint::replaceObservation(std::shared_ptr<Frame>observingFrame, size_t index)
{
	{
		std::lock_guard<std::mutex> lock(m_pointMutex);
		m_observingKeyframes[observingFrame] = index;
	}
	computeDescriptor();
	computeViewingDirection();
}

void MapPoint::removeObservation(std::shared_ptr<Frame>frame)
{
	{
		std::lock_guard<std::mutex> lock(m_pointMutex);

		if (m_observingKeyframes.find(frame) != m_observingKeyframes.end())
		{
			m_observingKeyframes.erase(frame);

			if (frame == m_referenceKeyframe)
			{
				if (m_observingKeyframes.size() > 0)
				{
					m_referenceKeyframe = m_observingKeyframes.begin()->first;
					m_referenceIndex = m_observingKeyframes.begin()->second;
					computeReferenceColor();
				}
				else {
					m_referenceKeyframe = nullptr;
					m_referenceIndex = -1;
					m_invalid = true;
				}
			}
		}
	}
}

Vector3f MapPoint::getViewingDirection()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_viewingDirection;
}

cv::Mat MapPoint::getDescriptor()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_descriptor;
}

std::map<std::shared_ptr<Frame>, size_t> MapPoint::getObservingKeyframes()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_observingKeyframes;
}

int MapPoint::getNumObserved()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_observingKeyframes.size();
}

bool MapPoint::fuse(std::shared_ptr<MapPoint>other)
{
	if (shared_from_this() == other)
		return false;

	std::map<std::shared_ptr<Frame>, size_t> m_obs;
	{
		std::lock_guard<std::mutex> lock(m_pointMutex);
		m_obs = m_observingKeyframes;
		m_observingKeyframes.clear();
		m_invalid = true;
	}

	// go over observations and update map point reference
	for (auto it = m_obs.begin(); it != m_obs.end(); it++)
	{

		std::shared_ptr<Frame>observingFrame = it->first;

		if (other->isObservedBy(observingFrame))
		{
			// since std::make_shared<MapPoint> is already observed, just delete association with this
			observingFrame->eraseAssociatedMapPoint(it->second);
		}
		else
		{
			// replace association by adding new point at index
			observingFrame->addAssociatedMapPoint(it->second, other);
			other->addObservationFast(observingFrame, it->second);
		}
	}
	other->computeDescriptor();
	other->computeViewingDirection();
	return true;
}

void MapPoint::computeViewingDirection()
{
	std::map<std::shared_ptr<Frame>, size_t> m_obs;
	Vector3f m_worldPos;
	std::shared_ptr<Frame>referenceFrame;
	{
		std::lock_guard<std::mutex> lock(m_pointMutex);
		m_obs = m_observingKeyframes;
		m_worldPos = m_worldPosition;
		referenceFrame = m_referenceKeyframe;
	}

	Vector3f viewDirSum = Vector3f::Zero();
	int n = 0;
	for (map<std::shared_ptr<Frame>, size_t>::iterator it = m_obs.begin(), end = m_obs.end(); it != end; it++)
	{
		std::shared_ptr<Frame>frame = it->first;
		Vector3f framePos = frame->getPose().block(0, 3, 3, 1);
		Vector3f viewDir = m_worldPos - framePos;
		viewDirSum += viewDir.normalized();
		n++;
	}

	// update min and max depth
	float worldDist = (m_worldPos - referenceFrame->getWorldPos()).norm();
	float levelScale = pow(1.2, referenceFrame->getKeypoint(m_obs[referenceFrame])->octave);
	float maxScale = pow(1.2, 7);

	{
		std::lock_guard<std::mutex> lock(m_pointMutex);
		m_viewingDirection = viewDirSum / n;
		m_maxDist = worldDist * levelScale;
		m_minDist = m_maxDist / maxScale;
	}
}

void MapPoint::computeDescriptor()
{
	std::map<std::shared_ptr<Frame>, size_t> m_obs;
	{
		std::lock_guard<std::mutex> lock(m_pointMutex);
		m_obs = m_observingKeyframes;
	}

	// for each descriptor in each frame compute distances to all other descriptors and take median value of all the distances. Decriptor with the smallest median distance is set to descriptor
	std::vector<cv::Mat> descriptors;
	for (map<std::shared_ptr<Frame>, size_t>::iterator it = m_obs.begin(), end = m_obs.end(); it != end; it++)
	{
		descriptors.push_back(it->first->getDescriptor(it->second));
	}

	std::vector<std::vector<float>> distances(descriptors.size(), std::vector<float>(descriptors.size()));

	for (int i = 0; i < descriptors.size(); i++)
	{
		cv::Mat currentDesc = descriptors[i];
		distances[i][i] = 0.;

		for (int j = i + 1; j < descriptors.size(); j++)
		{
			float dist = norm(currentDesc, descriptors[j], NORM_L2);
			distances[i][j] = dist;
			distances[j][i] = dist;
		}
	}

	int index = 0;
	int smallestDis = INT_MAX;
	for (int i = 0; i < descriptors.size(); i++)
	{
		std::vector<float> curDistances = distances[i];

		sort(curDistances.begin(), curDistances.end());
		int medianDist = curDistances[((descriptors.size() / 2.0) - 1)];

		if (medianDist < smallestDis)
		{
			smallestDis = medianDist;
			index = i;
		}
	}

	{
		std::lock_guard<std::mutex> lock(m_pointMutex);
		m_descriptor = descriptors[index].clone();
	}
}

bool MapPoint::isObservedBy(std::shared_ptr<Frame>frame)
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_observingKeyframes.find(frame) != m_observingKeyframes.end();
}

int MapPoint::getObservedIndex(std::shared_ptr<Frame>frame)
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	auto it = m_observingKeyframes.find(frame);
	if (it != m_observingKeyframes.end())
	{
		return it->second;
	}
	return -1;
}

std::shared_ptr<Frame>MapPoint::getReferenceFrame()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_referenceKeyframe;
}

Vector4uc MapPoint::getReferenceColor()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_referenceColor;
}

float MapPoint::getMinDistance()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_minDist;
}

float MapPoint::getMaxDistance()
{
	std::lock_guard<std::mutex> lock(m_pointMutex);
	return m_maxDist;
}

void MapPoint::erase()
{
	std::map<std::shared_ptr<Frame>, size_t> m_obs;
	{
		std::lock_guard<std::mutex> lock(m_pointMutex);
		m_obs = m_observingKeyframes;
		this->m_observingKeyframes.clear();
		m_invalid = true;
	}

	// go over observations and update map point reference
	for (auto it = m_obs.begin(); it != m_obs.end(); it++)
	{
		it->first->eraseAssociatedMapPoint(it->second);
	}
}

bool MapPoint::isInvalid()
{
	return m_invalid;
}

void MapPoint::computeReferenceColor()
{
	// Get the 2D keypoint position for this map point in the frame
	Point2f point = m_referenceKeyframe->getKeypoint(m_referenceIndex)->pt;
	Vector2f keypointPosition = Vector2f(point.x, point.y);
	// Access the color image from the frame
	cv::Mat colorImage = m_referenceKeyframe->getColor();
	// Ensure the keypoint position is within the image bounds
	if (keypointPosition.x() >= 0 && keypointPosition.x() < colorImage.cols &&
		keypointPosition.y() >= 0 && keypointPosition.y() < colorImage.rows)
	{
		// Retrieve the color at the keypoint position
		// pixel coordinates are subpixel, so we have to extract subpixel value
		cv::Mat interpolated;
		cv::getRectSubPix(colorImage, cv::Size(1, 1), cv::Point2f(keypointPosition.x(), keypointPosition.y()), interpolated);
		cv::Vec3b color = interpolated.at<cv::Vec3b>(0, 0);
		// Convert the color to Vector4uc format and return
		m_referenceColor = Vector4uc(color[0], color[1], color[2], 255); // Assuming full opacity
	}
	else
	{
		// Return a default color (e.g., black) if the position is out of bounds
		m_referenceColor = Vector4uc(0, 0, 0, 128);
	}
}
