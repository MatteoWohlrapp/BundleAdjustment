#include "BundleAdjustment.h"

using namespace std;
using namespace cv;

void BundleAdjustment::run()
{
	// Load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.init(datasetPath, datasetName))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return;
	}

	FeatureProcessor featureProcessor;
	if (!featureProcessor.init(featureType, DISPLAY_MATCHES))
	{
		std::cout << "Failed to initialize the feature processor!" << std::endl;
		return;
	}

	bool initialized = false;
	bool hasReferenceFrame = false;

	Visualizer *visualizer;

	if (displayPointCloud)
	{
		visualizer = new Visualizer(outputPath,map, false, true);
	}

	// use new to make an object with dynamic storage duration regardless of the scope
	std::shared_ptr<Frame>initialFrame = nullptr;
	std::shared_ptr<Frame>lastFrame = nullptr;
	Matrix4f lastLastPose = Matrix4f::Identity();

	int frameCounter = -1;
	for (int i = 0; sensor.processNextFrame() && i < processedFrames; i++)
	{
		frameCounter++;
		cv::Mat colorImg = cv::Mat(sensor.getColorImageHeight(), sensor.getColorImageWidth(), CV_8UC4, sensor.getColorRGBX());

		if (DISPLAY_COLOR)
		{
			cv::namedWindow("Color Image", cv::WindowFlags::WINDOW_AUTOSIZE);
			if (!colorImg.data)
			{
				printf("No color image data \n");
			}
			else
			{
				cv::Mat convImg;
				cv::cvtColor(colorImg, convImg, cv::COLOR_BGRA2RGBA);
				cv::imshow("Color Image", convImg);
			}
			cv::waitKey();
		}

		cv::Mat depthImg = cv::Mat(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32FC1, sensor.getDepth());

		if (DISPLAY_DEPTH)
		{
			cv::namedWindow("Depth Image", cv::WindowFlags::WINDOW_AUTOSIZE);
			if (!depthImg.data)
			{
				printf("No depth image data \n");
			}
			else
			{
				cv::Mat convImg;
				depthImg.convertTo(convImg, CV_8UC1);
				convImg *= 10.0;
				cv::imshow("Depth Image", convImg);
			}
			cv::waitKey();
		}
		std::shared_ptr<Frame>currentFrame = std::make_shared<Frame>(colorImg, depthImg, sensor.getTrajectory(), sensor.getColorIntrinsics(), &featureProcessor, frameCounter, sensor.getTimeStamp());
		currentFrame->init();
		
		if (!initialized)
		{
			if (!hasReferenceFrame)
			{
				//cout << "Setting frame" << endl;
				// create initializer and set first frame
				initialFrame = currentFrame;
				initialFrame->setPose(Matrix4f::Identity());
				initializer->setRefFrame(initialFrame);
				hasReferenceFrame = true;
				frameCounter = 0;
			}
			else
			{
				std::vector<cv::DMatch> matches = featureProcessor.matchFeatures(initialFrame, currentFrame);

				if (matches.size() > 100)
				{
					//cout << "Initializing" << endl;
					initialized = initializer->initialize(currentFrame, matches);
				}

				if (!initialized)
				{
					//cout << "Resetting Initializer" << endl;
					hasReferenceFrame = false;
					frameCounter = -1;
				}
				else
				{
					cout << "-------------------------   Initialized at frame " << i << "    -------------------------" << endl;
					/*
					SimpleMesh initMesh = SimpleMesh(facesType);
					initMesh.createMesh(map);
					initMesh.writeMesh(std::string("mesh_init.off"));*/
				}
				lastLastPose = initialFrame->getPose();
			}
			lastFrame = currentFrame;
			continue;
		}

		// Match features
		std::vector<DMatch> matches = featureProcessor.matchFeatures(lastFrame, currentFrame);

		int associatedPoints = 0;
		// Iterate through all matches to find existing 3D points and add references to std::make_shared<Frame>
		for (auto &match : matches)
		{
			std::shared_ptr<MapPoint>mapPoint = lastFrame->getMapPoint(match.queryIdx);
			if (mapPoint && !lastFrame->isOutlier(match.queryIdx))
			{
				// Keypoint was already matched in previous frame, so 3D position exists
				float descDist = norm(currentFrame->getDescriptor(match.trainIdx), mapPoint->getDescriptor(), NORM_L2);

				if (descDist < 0.2)
				{
					// Add frame as reference to mapPoint
					mapPoint->addObservation(currentFrame, match.trainIdx);
					// Add mapPoint as reference to frame
					currentFrame->addAssociatedMapPoint(match.trainIdx, mapPoint);
					associatedPoints++;
				}
			}
		}

		// Set initial pose based on constant speed assumption
		currentFrame->setPose(lastFrame->getPose()* lastLastPose.inverse()* lastFrame->getPose());

		// Estimate pose of currentFrame
		sfmHelper->estimatePose(currentFrame,lastFrame,&matches);

		float associatedRatio = associatedPoints / (float)currentFrame->getKeypointCount();
		//cout << "Ratio associated: " << associatedRatio << endl;

		if (associatedRatio <= 0.0001) {
			cout << "-------------------------       Tracking failed        -------------------------" << endl;
			break;
		}
		if (associatedRatio <= 0.1)
		{
			sfmHelper->cullRecentMapPoints(currentFrame, map);

			// After we set the pose for the new camera, we can triangulate points that were not recorded yet
			std::vector<DMatch> triangulateMatches;
			for (auto &match : matches)
			{
				std::shared_ptr<MapPoint>mapPoint = lastFrame->getMapPoint(match.queryIdx);
				if (!mapPoint && !lastFrame->isOutlier(match.queryIdx))
				{
					// Match does not exist, so we need to create a std::make_shared<MapPoint> and add it to both frames
					// Triangulate new 3D point
					if (!lastFrame->isOutlier(match.queryIdx) && !currentFrame->isOutlier(match.trainIdx))
					{
						// triangulate only if both points are no outliers
						triangulateMatches.push_back(match);
					}
				}
			}

			if (triangulateMatches.size() > 0)
			{
				sfmHelper->triangulatePoints(lastFrame, currentFrame, &triangulateMatches, map, true);
			}

			// Add frame to scene
			map->addKeyFrame(currentFrame);

			// create covisibility graph
			currentFrame->updateCovisibilityGraph();

			// Search points in neighbor keyframes
			sfmHelper->searchInNeighbors(currentFrame, &featureProcessor, map);

			if (localBA)
			{
				localOptimizer->setNbOfIterations(1);
				localOptimizer->setNbOfMaxItPerBA(10);
				localOptimizer->optimizeCamerasAndMapPoints(map, currentFrame,true);
			}
			else
			{
				optimizer->setNbOfIterations(1);
				optimizer->setNbOfMaxItPerBA(10);
				optimizer->optimizeCamerasAndMapPoints(map, true,frameCounter);
			}

			if (cullFrames)
			{
				sfmHelper->cullRedundantKeyframes(currentFrame, map);
			}
		}

		lastLastPose = lastFrame->getPose();
		if (!lastFrame->isKeyFrame()) {
			lastFrame->erase();
		}
		lastFrame = currentFrame;
	}

	/*
	cout << "PoseFrame0 \n"
		 << map->getKeyFrames()[0]->getPose() << endl;
	*/

	// after processing each frame call the GlobalBAOptimizer
	optimizer->setNbOfIterations(3);
	optimizer->setNbOfMaxItPerBA(100);
	optimizer->optimizeCamerasAndMapPoints(map, true, frameCounter);

	//erase all remaining outliers
	sfmHelper->eraseOutlier(map, 0);

	/*
	// print out and calculate difference between all relative poses
	for (int i = 1; i < map->getKeyFrameCount(); i++)
	{
		std::shared_ptr<Frame>kf = map->getKeyFrames()[i];
		std::shared_ptr<Frame>kf_before = map->getKeyFrames()[i - 1];
		// cout << "pose \n" << kf->getPose() << endl << "gtpose \n" << kf->getGTPose() << endl;
		Matrix4f relpose = (kf_before->getPose().inverse() * kf->getPose());
		Matrix4f relGTpose = (kf_before->getGTPose().inverse() * kf->getGTPose());
		auto diffnorm = (relpose - relGTpose).norm();
		// cout << i << " relpose \n" << relpose << i << " relGTpose \n" << relGTpose;
		cout << i << " posedifference " << diffnorm << endl;
	}*/

	if (trajectory)
	{
		std::cout << "Writing trajectory to file" << std::endl;
		std::ofstream f(outputPath + "_estimatedPoses.txt");
		if (f.is_open()) {
			std::vector<std::shared_ptr<Frame>> frames = map->getKeyFrames();
			for (int i = 0; i < map->getKeyFrameCount(); i++)
			{
				std::shared_ptr<Frame> cur = frames[i];
				Matrix4f pose = cur->getPose();
				Matrix3f Rot = pose.block(0, 0, 3, 3);
				// camera center
				Vector3f t = pose.block(0, 3, 3, 1);
				Eigen::Quaternionf q(Rot);
				// std::cout << t << std::endl;
				f << setprecision(4) << fixed << cur->getTimeStamp() << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
			}
			f.close();
		}
	}

	std::cout << "Creating mesh" << std::endl;
	mesh->createMesh(map);
	std::cout << "Writing mesh to file: " << outputPath << std::endl;
	mesh->writeMesh(outputPath + "_mesh.off");

	if (displayPointCloud)
	{
		delete visualizer;
	}

	std::cout << "Destroying windows" << std::endl;
	cv::destroyAllWindows();
}
