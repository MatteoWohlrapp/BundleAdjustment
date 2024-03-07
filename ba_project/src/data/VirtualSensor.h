#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "../utils/Eigen.h"
#include "FreeImageHelper.h"

typedef unsigned char BYTE;

enum DatasetName
{
	FreiburgXYZ,
	FreiburgTeddy,
	ReplicaRoom
};

// reads sensor files
class VirtualSensor
{
public:
	VirtualSensor() : m_currentIdx(-1), m_increment(1) {}

	~VirtualSensor()
	{
		SAFE_DELETE_ARRAY(m_depthFrame);
		SAFE_DELETE_ARRAY(m_colorFrame);
	}

	bool init(const std::string &datasetDir, const DatasetName datasetname)
	{
		m_baseDir = datasetDir;

		if (datasetname == DatasetName::FreiburgXYZ || datasetname == DatasetName::FreiburgTeddy)
		{
			return loadFreiburg(datasetname);
		}
		else if (datasetname == DatasetName::ReplicaRoom)
		{
			return loadReplica();
		}

		return true;
	}
	bool loadReplica()
	{
		// fill vectors with filenames
		m_filenameDepthImages.clear();
		m_filenameColorImages.clear();
		for (int i = 0; i < 2000; i++)
		{
			std::stringstream ss;
			ss << std::setw(6) << std::setfill('0') << i;
			std::string fnameDepth = std::string("results/") + "depth" + ss.str() + ".png";
			m_filenameDepthImages.push_back(fnameDepth);
			// std::cout << fnameDepth;

			std::stringstream ssRGB;
			ssRGB << std::setw(6) << std::setfill('0') << i;
			std::string fnameRGB = std::string("results/") + "frame" + ssRGB.str() + ".jpg";
			m_filenameColorImages.push_back(fnameRGB);
			// std::cout << fnameRGB;

			// set increasing timestamps
			m_depthImagesTimeStamps.push_back(i + 1);
			m_colorImagesTimeStamps.push_back(i + 1);
		}

		// Read tracking poses from traj.txt
		m_trajectory.clear();
		std::ifstream file(m_baseDir + "traj.txt", std::ios::in);
		if (!file.is_open())
			return false;

		float timestamp_value = 1.0f;
		while (file.good())
		{
			Eigen::Matrix4f traj;
			traj.setIdentity();
			file >> traj(0) >> traj(1) >> traj(2) >> traj(3) >> traj(4) >> traj(5) >> traj(6) >> traj(7) >> traj(8) >> traj(9) >> traj(10) >> traj(11) >> traj(12) >> traj(13) >> traj(14) >> traj(15);

			// traj = traj.inverse().eval(); // could be a GT trajectory convention from Freiburg datasets?
			m_trajectory.push_back(traj);

			// increasing timestamps are set with an increment of 0.01 (similar to the Freiburg dataset)
			m_trajectoryTimeStamps.push_back(timestamp_value);
			timestamp_value += 0.01f;
		}
		m_trajectory.pop_back(); // because of the last empty line of the traj.txt file
		file.close();

		if (m_filenameDepthImages.size() != m_filenameColorImages.size())
			return false;

		// Image resolutions from cam_params.json
		m_colorImageWidth = 1200;
		m_colorImageHeight = 680;
		m_depthImageWidth = 1200;
		m_depthImageHeight = 680;

		// Intrinsics from cam_params.json
		//"fx": 600.0,
		//"fy": 600.0,
		//"cx" : 599.5,
		//"cy" : 339.5,
		//"scale" : 6553.5
		m_depthImageScale = 6553.5;
		m_colorIntrinsics << 600.0f, 0.0f, 599.5f,
			0.0f, 600.0f, 339.5f,
			0.0f, 0.0f, 1.0f;

		m_depthIntrinsics = m_colorIntrinsics;

		m_colorExtrinsics.setIdentity();
		m_depthExtrinsics.setIdentity();

		m_depthFrame = new float[m_depthImageWidth * m_depthImageHeight];
		for (unsigned int i = 0; i < m_depthImageWidth * m_depthImageHeight; ++i)
			m_depthFrame[i] = 0.5f;

		m_colorFrame = new BYTE[4 * m_colorImageWidth * m_colorImageHeight];
		for (unsigned int i = 0; i < 4 * m_colorImageWidth * m_colorImageHeight; ++i)
			m_colorFrame[i] = 255;

		m_currentIdx = -1;
		return true;
	}

	bool loadFreiburg(const DatasetName datasetname)
	{

		// Read filename lists
		if (!readFileList(m_baseDir + "depth.txt", m_filenameDepthImages, m_depthImagesTimeStamps))
			return false;
		if (!readFileList(m_baseDir + "rgb.txt", m_filenameColorImages, m_colorImagesTimeStamps))
			return false;

		// Read tracking
		if (!readTrajectoryFile(m_baseDir + "groundtruth.txt", m_trajectory, m_trajectoryTimeStamps))
			return false;

		if (datasetname == DatasetName::FreiburgTeddy)
		{
			m_filenameColorImages.pop_back(); // because there is one entry too much in the rgb.txt file
		}
		if (m_filenameDepthImages.size() != m_filenameColorImages.size())
			return false;

		// Image resolutions
		m_colorImageWidth = 640;
		m_colorImageHeight = 480;
		m_depthImageWidth = 640;
		m_depthImageHeight = 480;

		// Intrinsics
		m_colorIntrinsics << 525.0f, 0.0f, 319.5f,
			0.0f, 525.0f, 239.5f,
			0.0f, 0.0f, 1.0f;
		// depth images are scaled by 5000 (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
		m_depthImageScale = 5000.0f;

		m_depthIntrinsics = m_colorIntrinsics;

		m_colorExtrinsics.setIdentity();
		m_depthExtrinsics.setIdentity();

		m_depthFrame = new float[m_depthImageWidth * m_depthImageHeight];
		for (unsigned int i = 0; i < m_depthImageWidth * m_depthImageHeight; ++i)
			m_depthFrame[i] = 0.5f;

		m_colorFrame = new BYTE[4 * m_colorImageWidth * m_colorImageHeight];
		for (unsigned int i = 0; i < 4 * m_colorImageWidth * m_colorImageHeight; ++i)
			m_colorFrame[i] = 255;

		m_currentIdx = -1;

		return true;
	}

	bool processNextFrame()
	{
		if (m_currentIdx == -1)
			m_currentIdx = 0;
		else
			m_currentIdx += m_increment;

		if ((unsigned int)m_currentIdx >= (unsigned int)m_filenameColorImages.size())
			return false;

		std::cout << "------------------------- Processing Frame [" << m_currentIdx << " | " << m_filenameColorImages.size() << "] -------------------------" << std::endl;

		FreeImageB rgbImage;
		bool loadedimage = rgbImage.LoadImageFromFile(m_baseDir + m_filenameColorImages[m_currentIdx]);
		memcpy(m_colorFrame, rgbImage.data, 4 * m_depthImageWidth * m_depthImageHeight);

		FreeImageU16F dImage;
		dImage.LoadImageFromFile(m_baseDir + m_filenameDepthImages[m_currentIdx]);

		for (unsigned int i = 0; i < m_depthImageWidth * m_depthImageHeight; ++i)
		{
			if (dImage.data[i] == 0)
				m_depthFrame[i] = MINF;
			else
				m_depthFrame[i] = dImage.data[i] * 1.0f / m_depthImageScale;
		}

		// find transformation (simple nearest neighbor, linear search)
		double timestamp = m_depthImagesTimeStamps[m_currentIdx];
		double min = std::numeric_limits<double>::max();
		int idx = 0;
		for (unsigned int i = 0; i < m_trajectory.size(); ++i)
		{
			double d = abs(m_trajectoryTimeStamps[i] - timestamp);
			if (min > d)
			{
				min = d;
				idx = i;
			}
		}
		m_currentTrajectory = m_trajectory[idx];

		return true;
	}

	unsigned int getCurrentFrameCnt()
	{
		return (unsigned int)m_currentIdx;
	}

	// get current color data
	BYTE *getColorRGBX()
	{
		return m_colorFrame;
	}

	// get current depth data
	float *getDepth()
	{
		return m_depthFrame;
	}

	// color camera info
	Eigen::Matrix3f getColorIntrinsics()
	{
		return m_colorIntrinsics;
	}

	Eigen::Matrix4f getColorExtrinsics()
	{
		return m_colorExtrinsics;
	}

	unsigned int getColorImageWidth()
	{
		return m_colorImageWidth;
	}

	unsigned int getColorImageHeight()
	{
		return m_colorImageHeight;
	}

	// depth (ir) camera info
	Eigen::Matrix3f getDepthIntrinsics()
	{
		return m_depthIntrinsics;
	}

	Eigen::Matrix4f getDepthExtrinsics()
	{
		return m_depthExtrinsics;
	}

	unsigned int getDepthImageWidth()
	{
		return m_depthImageWidth;
	}

	unsigned int getDepthImageHeight()
	{
		return m_depthImageHeight;
	}

	// get current trajectory transformation
	Eigen::Matrix4f getTrajectory()
	{
		return m_currentTrajectory;
	}

	// get timestamp
	double getTimeStamp()
	{
		return m_trajectoryTimeStamps[m_currentIdx];
	}

private:
	bool readFileList(const std::string &filename, std::vector<std::string> &result, std::vector<double> &timestamps)
	{
		std::ifstream fileDepthList(filename, std::ios::in);
		if (!fileDepthList.is_open())
			return false;
		result.clear();
		timestamps.clear();
		std::string dump;
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		while (fileDepthList.good())
		{
			double timestamp;
			fileDepthList >> timestamp;
			std::string filename;
			fileDepthList >> filename;
			if (filename == "")
				break;
			timestamps.push_back(timestamp);
			result.push_back(filename);
		}
		fileDepthList.close();
		return true;
	}

	bool readTrajectoryFile(const std::string &filename, std::vector<Eigen::Matrix4f> &result,
							std::vector<double> &timestamps)
	{
		std::ifstream file(filename, std::ios::in);
		if (!file.is_open())
			return false;
		result.clear();
		std::string dump;
		std::getline(file, dump);
		std::getline(file, dump);
		std::getline(file, dump);

		while (file.good())
		{
			double timestamp;
			file >> timestamp;
			Eigen::Vector3f translation;
			file >> translation.x() >> translation.y() >> translation.z();
			Eigen::Quaternionf rot;
			file >> rot;

			Eigen::Matrix4f transf;
			transf.setIdentity();
			transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
			transf.block<3, 1>(0, 3) = translation;

			if (rot.norm() == 0)
				break;

			transf = transf.inverse().eval();

			timestamps.push_back(timestamp);
			result.push_back(transf);
		}
		file.close();
		return true;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// current frame index
	int m_currentIdx;

	int m_increment;

	// frame data
	float *m_depthFrame;
	BYTE *m_colorFrame;
	Eigen::Matrix4f m_currentTrajectory;

	// color camera info
	Eigen::Matrix3f m_colorIntrinsics;
	Eigen::Matrix4f m_colorExtrinsics;
	unsigned int m_colorImageWidth;
	unsigned int m_colorImageHeight;

	// depth (ir) camera info
	Eigen::Matrix3f m_depthIntrinsics;
	Eigen::Matrix4f m_depthExtrinsics;
	unsigned int m_depthImageWidth;
	unsigned int m_depthImageHeight;
	float m_depthImageScale;

	// base dir
	std::string m_baseDir;
	// filenamelist depth
	std::vector<std::string> m_filenameDepthImages;
	std::vector<double> m_depthImagesTimeStamps;
	// filenamelist color
	std::vector<std::string> m_filenameColorImages;
	std::vector<double> m_colorImagesTimeStamps;

	// trajectory
	std::vector<Eigen::Matrix4f> m_trajectory;
	std::vector<double> m_trajectoryTimeStamps;
};
