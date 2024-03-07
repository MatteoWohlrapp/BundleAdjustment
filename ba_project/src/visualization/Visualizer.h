#pragma once

#include <thread>
#include <iostream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/io/ply_io.h>
#include "../model/SceneMap.h"

class Visualizer
{
public:
	Visualizer(std::string outputFilePath, SceneMap *map, bool displayViewingNormals = false, bool displayOutliers = false);
	~Visualizer();

	void processEvents();

private:
	void onRender();

	bool m_stop;
	bool m_displayOutliers;
	bool m_displayViewingNormals;
	SceneMap *m_map;
	pcl::visualization::PCLVisualizer::Ptr m_viewer;
	std::thread *m_visualizationThread;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr inlierCloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr outlierCloud;
	pcl::PointCloud<pcl::Normal>::Ptr inlierNormalCloud;
	std::string m_outputFilePath;
};
