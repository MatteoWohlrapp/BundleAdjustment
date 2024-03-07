#include <iostream>
#include <string>
#include <map>
#include "./visualization/SimpleMesh.h"
#include "./ba/BundleAdjustment.h"
#include "./ba/SfMHelper.h"
#include "./ba/Initializer.h"
#include "./metrics/ReconstructionError.h"
#include "./ba/FeatureProcessor.h"

using namespace std;
using namespace cv;

// Define default values for your options
const std::map<std::string, InitType> initTypeMap = {{"standard", Standard}, {"gtdepth", GTDepth}};
const std::map<std::string, EstimationType> estimationTypeMap = { {"pnp", PNP}, {"ba", BA}, {"2d2d", EssentialOrHomography} };
const std::map<std::string, FacesType> facesTypeMap = {{"poisson", Poisson}, {"greedy", GreedyProjectionTriangulation}, {"none", None}};
const std::map<std::string, std::pair<DatasetName, std::string>> datasetNameMap = {
	{"FreiburgXYZ", {FreiburgXYZ, "../../data/rgbd_dataset_freiburg1_xyz/"}},
	{"FreiburgTeddy", {FreiburgTeddy, "../../data/rgbd_dataset_freiburg1_teddy/"}},
	{"ReplicaRoom", {ReplicaRoom, "../../data/Replica/room0/"}}};
const std::map<std::string, FeatureType> featureTypeMap = {
	{"harriscorner", HarrisCorner},
	{"shitomasi", ShiTomasi},
	{"surf", SURF}};

void printHelp()
{
	std::cout << "Usage: program [options]\n";
	std::cout << "Options:\n";
	std::cout << "  --help ------------------------------------------------ Print this help message\n";
	std::cout << "  --init-type [standard|gtdepth] ------------------------ Set the initialization type\n";
	std::cout << "  --estimation [pnp|ba|2d2d] ---------------------------- Set the frame to frame estimation type\n";
	std::cout << "  --faces-type [poisson|greedy|none] -------------------- Set the faces type\n";
	std::cout << "  --dataset-path [path] --------------------------------- Set a custom dataset path (overrides standard path)\n";
	std::cout << "  --dataset-name [FreiburgXYZ|FreiburgTeddy|ReplicaRoom]  Set the dataset name which selects the predefined path\n";
	std::cout << "  --output-path [path] ---------------------------------- Set the output path\n";
	std::cout << "  --local-ba -------------------------------------------- Set local bundle adjustment\n";
	std::cout << "  --frames [number] ------------------------------------- Set the number of frames to process\n";
	std::cout << "  --reconstruction-error -------------------------------- Compute reconstruction error. Make sure, you have Replica at the correct file path: ../../data/Replica/Replica/room0/.\n";
	std::cout << "  --trajectory ------------------------------------------ Set if you want to save the trajectory to a file\n";
	std::cout << "  --display-pointcloud ---------------------------------- Display the pointcloud in real time\n";
	std::cout << "  --cull-frames ----------------------------------------- Cull frames\n";
}

int main(int argc, char *argv[])
{
	// Default values
	InitType initType = Standard;
	EstimationType estimation = BA;
	FacesType facesType = None;
	std::string customDatasetPath;
	bool customPathProvided = false;
	DatasetName selectedDataset = ReplicaRoom;
	std::string datasetPath = datasetNameMap.find("ReplicaRoom")->second.second;
	std::string outputPath = "../output/";
	bool localBA = false;
	int processedFrames = 2000; // Default number of frames
	bool computeReconstructionError = false;
	std::string reconstructionErrorDatasetPath;
	FeatureType featureType = FeatureType::SURF;
	bool trajectory = false;
	bool displayPointCloud = false;
	bool cullFrames = false;

	// Parse command-line arguments
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];

		if (arg == "--help")
		{
			printHelp();
			return 0;
		}
		else if (arg == "--init-type" && i + 1 < argc)
		{
			auto it = initTypeMap.find(argv[++i]);
			if (it != initTypeMap.end())
			{
				initType = it->second;
			}
			else
			{
				std::cerr << "Unknown init type: " << argv[i] << std::endl;
				return 1;
			}
		}
		else if (arg == "--estimation" && i + 1 < argc)
		{
			auto it = estimationTypeMap.find(argv[++i]);
			if (it != estimationTypeMap.end())
			{
				estimation = it->second;
			}
			else
			{
				std::cerr << "Unknown estimation type: " << argv[i] << std::endl;
				return 1;
			}
		}
		else if (arg == "--faces-type" && i + 1 < argc)
		{
			auto it = facesTypeMap.find(argv[++i]);
			if (it != facesTypeMap.end())
			{
				facesType = it->second;
			}
			else
			{
				std::cerr << "Unknown estimation type: " << argv[i] << std::endl;
				return 1;
			}
		}
		else if (arg == "--dataset-name" && i + 1 < argc)
		{
			auto it = datasetNameMap.find(argv[++i]);
			if (it != datasetNameMap.end())
			{
				selectedDataset = it->second.first;
				datasetPath = it->second.second;
			}
			else
			{
				std::cerr << "Unknown dataset name: " << argv[i] << std::endl;
				return 1;
			}
		}
		else if (arg == "--dataset-path" && i + 1 < argc)
		{
			customDatasetPath = argv[++i];
			customPathProvided = true;
		}
		else if (arg == "--output-path" && i + 1 < argc)
		{
			outputPath = argv[++i];
		}
		else if (arg == "--local-ba")
		{
			localBA = !localBA;
		}
		else if (arg == "--frames" && i + 1 < argc)
		{
			processedFrames = std::stoi(argv[++i]);
		}
		else if (arg == "--reconstruction-error")
		{
			computeReconstructionError = !computeReconstructionError;
 		}
		else if (arg == "--trajectory")
		{
			trajectory = !trajectory;
		}
		else if (arg == "--display-pointcloud")
		{
			displayPointCloud = !displayPointCloud;
		}
		else if (arg == "--cull-frames")
		{
			cullFrames = !cullFrames;
		}
		else
		{
			std::cerr << "Unknown option: " << arg << std::endl;
			printHelp();
			return 1;
		}
	}

	if (customPathProvided)
	{
		datasetPath = customDatasetPath;
	}

	std::stringstream ss;
	ss << "output_"
	   << (localBA ? "localBA" : "globalBA") << "_"
	   << "frames" << processedFrames << "_"
	   << "init" << initType << "_"
	   << "est" << estimation << "_"
	   << "faces" << facesType;

	// combine outputpath and stringstream
	outputPath = outputPath + ss.str();

	BundleAdjustment ba(featureType, initType, estimation, facesType, datasetPath, selectedDataset, outputPath, localBA, processedFrames, trajectory, displayPointCloud, cullFrames);
	ba.run();

	if (computeReconstructionError)
	{
		// calculating the reconstruction error on the replica dataset
		if (selectedDataset == DatasetName::ReplicaRoom)
		{
			ReconstructionError reconError = ReconstructionError(outputPath);
			double errorValue = reconError.computeReconstructionError(ba.getMap(), datasetPath + "../room0_mesh.ply");
			std::cout << "errorValue " << errorValue << std::endl;
		}
		else
		{
			std::cout << "Reconstruction error can only be computed on the Replica dataset" << std::endl;
		}
	}

	return 0;
}