#include "Visualizer.h"

Visualizer::Visualizer(std::string outputFilePath, SceneMap *map, bool displayViewingNormals, bool displayOutliers)
	: m_map(map), m_stop(false), m_displayOutliers(displayOutliers), m_displayViewingNormals(displayViewingNormals), m_outputFilePath(outputFilePath)
{
	// start thread
	// visualizer->processEvents()
	m_visualizationThread = new std::thread(&Visualizer::processEvents, this);
}

Visualizer::~Visualizer()
{
	m_stop = true;
	m_visualizationThread->join();
}

void Visualizer::processEvents()
{
	m_viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
	m_viewer->setBackgroundColor(0.02, 0.02, 0.02);
	m_viewer->addCoordinateSystem(0.25);
	// m_viewer->initCameraParameters();
	m_viewer->setCameraPosition(0.0, -1.0, -5.0, 0.0, -1., 0);
	m_viewer->setCameraClipDistances(0.1, 1.0);
	inlierCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	outlierCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	inlierNormalCloud = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);

	while (!m_viewer->wasStopped() && !m_stop)
	{
		m_viewer->spinOnce(100);
		onRender();
		std::this_thread::sleep_for(100ms);
	}

	cout << "Writing pcl cloud to file..." << endl;

	if (m_displayOutliers)
	{

		for (int i = 0; i < outlierCloud->size(); i++)
		{
			inlierCloud->push_back(pcl::PointXYZRGB(outlierCloud->points[i].x, outlierCloud->points[i].y, outlierCloud->points[i].z, 255, 0, 0));
		}
		pcl::io::savePLYFile((m_outputFilePath + std::string("_PCL_cloud_final_with_outliers.ply")), *inlierCloud);
	}
	else
	{
		pcl::io::savePLYFile((m_outputFilePath + std::string("_PCL_cloud_final.ply")), *inlierCloud);
	}
}

void Visualizer::onRender()
{
	if (!m_viewer->wasStopped())
	{
		// delete old data
		m_viewer->removePointCloud("inliers");
		m_viewer->removePointCloud("outliers");
		m_viewer->removePointCloud("inlierNormals");
		m_viewer->removeAllShapes();

		// gather data from sceneMap
		vector<std::shared_ptr<MapPoint>> map_points = m_map->getMapPoints();

		// clear inlier and outlier clouds
		inlierCloud->clear();
		outlierCloud->clear();
		inlierNormalCloud->clear();

		if (map_points.size() > 0)
		{
			// add mappoints
			for (int idx = 0; idx < map_points.size(); ++idx)
			{
				if (map_points[idx]) {
					Vector3f position = map_points[idx]->getPosition();
					Vector3f viewNormal = map_points[idx]->getViewingDirection() * -1.;
					// go through map of all observing frames for the map point and check if they are outliers
					bool isOutlier = false;
					for (auto& pair : (map_points[idx]->getObservingKeyframes())) {
						std::shared_ptr<Frame> frame = pair.first;
						size_t indexInFrame = pair.second;
						if (frame->isOutlier(indexInFrame))
						{
							isOutlier = true;
							break;
						}
					}

					// check if outlier and positions for x, y, z smaller than 1
					if (!isOutlier)
					{
						// get color from reference frame
						Vector4uc color = map_points[idx]->getReferenceColor();

						inlierCloud->push_back(pcl::PointXYZRGB(position.x(), position.y(), position.z(), color.x(), color.y(), color.z()));
						inlierNormalCloud->push_back(pcl::Normal(viewNormal.x(), viewNormal.y(), viewNormal.z()));
					}
					else
					{
						outlierCloud->push_back(pcl::PointXYZ(position.x(), position.y(), position.z()));
					}
				}
			}

			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(inlierCloud);
			m_viewer->addPointCloud<pcl::PointXYZRGB>(inlierCloud, rgb, "inliers");
			m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inliers");

			if (m_displayViewingNormals)
			{
				m_viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(inlierCloud, inlierNormalCloud, 1, 0.007, "inlierNormals");
			}

			if (m_displayOutliers)
			{
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> singleColor(outlierCloud, 255, 0, 0);
				m_viewer->addPointCloud<pcl::PointXYZ>(outlierCloud, singleColor, "outliers");
				m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "outliers");
			}
		}

		// add cameras
		vector<std::shared_ptr<Frame>> keyFrames = m_map->getKeyFrames();

		if (keyFrames.size() > 0)
		{
			Matrix4f worldPose = keyFrames[0]->getGTPose().inverse();

			Eigen::Matrix4f swapMatrix = Eigen::Matrix4f::Identity();

			// Swap the dimensions: x -> y, -y -> z, -z -> x
			swapMatrix(0, 0) = 0.0;
			swapMatrix(0, 1) = -1.0;
			swapMatrix(0, 2) = 0.;
			swapMatrix(1, 0) = -1.0;
			swapMatrix(1, 1) = 0.0;
			swapMatrix(1, 2) = 0.;
			swapMatrix(2, 0) = 0.;
			swapMatrix(2, 1) = 0.0;
			swapMatrix(2, 2) = 1.0;

			//approx scale
			float ourDist = ((keyFrames[0]->getPose() * Vector3f::Zero().homogeneous()).hnormalized() - (keyFrames[1]->getPose() * Vector3f::Zero().homogeneous()).hnormalized()).norm();
			float gtDist = ((keyFrames[0]->getGTPose() * Vector3f::Zero().homogeneous()).hnormalized() - (keyFrames[1]->getGTPose() * Vector3f::Zero().homogeneous()).hnormalized()).norm();
			float ratio = ourDist / gtDist;

			int maxInd = keyFrames.back()->getID();
			for (int i = 0; i < keyFrames.size(); i++)
			{
				Matrix4f pose = keyFrames[i]->getPose();

				Vector3f pos = (pose * Vector3f::Zero().homogeneous()).hnormalized();
				Vector3f forward = (pose * Vector4f(0.0, 0.0, 0.05, 1.0)).hnormalized();
				pcl::PointXYZ posPCL = pcl::PointXYZ(pos.x(), pos.y(), pos.z());
				pcl::PointXYZ forwardPCL = pcl::PointXYZ(forward.x(), forward.y(), forward.z());
				std::string id = std::to_string(keyFrames[i]->getID());
				double color = (float)keyFrames[i]->getID() / (float)maxInd;
				m_viewer->addArrow(forwardPCL, posPCL, 0.0, color, 0.0, false, id);

				Matrix4f poseGT = swapMatrix * worldPose * keyFrames[i]->getGTPose();
				Vector3f posGT = ratio * (poseGT * Vector3f::Zero().homogeneous()).hnormalized();
				Vector3f forwardGT = ratio * (poseGT * Vector4f(0.0, 0.0, 0.05 * 1/ratio, 1.0)).hnormalized();
				pcl::PointXYZ posPCLGT = pcl::PointXYZ(posGT.x(), posGT.y(), posGT.z());
				pcl::PointXYZ forwardPCLGT = pcl::PointXYZ(forwardGT.x(), forwardGT.y(), forwardGT.z());
				std::string idGT = std::to_string(keyFrames[i]->getID()) + "GT";
				m_viewer->addArrow(forwardPCLGT, posPCLGT, color,0.0 , 0.0, false, idGT);

			}
		}
	}
}
