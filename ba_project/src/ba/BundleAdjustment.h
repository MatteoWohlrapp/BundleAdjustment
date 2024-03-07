#pragma once

#define _HAS_STD_BYTE 0
#include "FeatureProcessor.h"
#include "../model/SceneMap.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "SfMHelper.h"
#include "../visualization/SimpleMesh.h"
#include "../visualization/Visualizer.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#define DISPLAY_COLOR 0
#define DISPLAY_DEPTH 0
#define DISPLAY_MATCHES 0

// Class to perform bundle adjustment
class BundleAdjustment
{
private:
    InitType initType;
    Initializer *initializer;
    EstimationType estimation;
    SfMHelper *sfmHelper;
    FacesType facesType;
    SimpleMesh *mesh;
    SceneMap *map;
    GlobalBAOptimizerAngles *optimizer;
    LocalBAOptimizerAngles* localOptimizer;
    std::string datasetPath;
    DatasetName datasetName;
    std::string outputPath;
    bool localBA;
    bool trajectory;
    int processedFrames;
    FeatureType featureType;
    bool displayPointCloud;
    bool cullFrames;

public:
    BundleAdjustment(FeatureType featureType, InitType initType, EstimationType estimation, FacesType facesType, std::string datasetPath, DatasetName datasetName, std::string outputPath, bool localBA, int processedFrames, bool trajectory, bool displayPointCloud, bool cullFrames) : featureType(featureType), initType(initType), estimation(estimation), facesType(facesType), datasetPath(datasetPath), datasetName(datasetName), outputPath(outputPath), localBA(localBA), processedFrames(processedFrames), trajectory(trajectory), displayPointCloud(displayPointCloud), cullFrames(cullFrames)
    {
        // create scene map
        optimizer = new GlobalBAOptimizerAngles();
        localOptimizer = new LocalBAOptimizerAngles();
        map = new SceneMap();
        sfmHelper = SfMHelper::getInstance();
        sfmHelper->setPoseEstimationType(estimation);
        initializer = new Initializer(nullptr, initType, map, sfmHelper);
        mesh = new SimpleMesh(facesType);
    }
    ~BundleAdjustment()
    {
        delete map;
        delete sfmHelper;
        delete initializer;
        delete mesh;
        delete optimizer;
        delete localOptimizer;
    };

    // Get the scene map
    SceneMap *getMap()
    {
        return map;
    }

    // Run bundle adjustment
    void run();
};