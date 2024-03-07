#include "Optimizer.h"

using namespace cv;

// the same functions for all Opimizers
void BAOptimizer::pruneCorrespondences(std::shared_ptr<Frame>frame, bool considerOutlier)
{

    int keypointCount = frame->getKeypointCount();
    Matrix4f extr = frame->getPose().inverse();
    Matrix3f intr = frame->getIntrinsics();

    int numOutlier = 0;
    int numInlier = 0;

    for (int i = 0; i < keypointCount; i++)
    {
        KeyPoint *keyPoint = frame->getKeypoint(i);
        std::shared_ptr<MapPoint>mapPoint = frame->getMapPoint(i);

        if (considerOutlier || !frame->isOutlier(i))
        {
            if (mapPoint)
            {
                // only process keypoints that have associated mappoints
                float chiThresh = 5.991;
                float sigmaFactor = 1.2;
                float invSigmaP = 1.0 / pow(1.2, keyPoint->octave);

                Vector2f keyPointPos;
                keyPointPos << keyPoint->pt.x, keyPoint->pt.y;

                Vector3f camspacePos = (extr * (mapPoint->getPosition().homogeneous())).hnormalized();

                // check if point is in front of camera
                if (camspacePos.z() <= 0)
                {
                    frame->setOutlier(i);
                    numOutlier++;
                    continue;
                }

                // check if depth bounds mappoint is inside the scale pyramid of the points reference frame
                float worldDist = (mapPoint->getPosition() - frame->getWorldPos()).norm();
                float minDist = mapPoint->getMinDistance();
                float maxDist = mapPoint->getMaxDistance();
                if (worldDist > maxDist || worldDist < minDist)
                {
                    frame->setOutlier(i);
                    numOutlier++;
                    continue;
                }

                Vector2f projPos = (intr * camspacePos).hnormalized();

                float squaredDist = (projPos - keyPointPos).norm();
                float chiSquare = squaredDist * invSigmaP;

                if (chiSquare > chiThresh)
                {
                    // mark keypoint as outlier for now and therefore exclude it from further optimizations
                    frame->setOutlier(i);
                    numOutlier++;
                    continue;
                }

                // mark keypoint as inlier and therefore include it in further optimizations
                frame->setInlier(i);
                numInlier++;
            }
        }
        else
        {
            numOutlier++;
        }
    }

    // cout << "Set " << numOutlier << " outliers and " << numInlier << " inliers after Optimization" << endl;
}
void BAOptimizer::configureSolver(ceres::Solver::Options &options)
{
    // Ceres options.
    options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    options.linear_solver_type = ceres::DENSE_SCHUR; // DENSE_SCHUR advised; Ceres tutorial says that it is "well large for DENSE_QR anyways"
    options.minimizer_progress_to_stdout = 1;
    options.max_num_iterations = m_nMaxItPerBA;
    options.num_threads = 4;
    options.logging_type = ceres::SILENT;
}

// GlobalBAOptimizer with optimizing <3, 4> matrizes for camera poses
void GlobalBAOptimizer::optimizeCamerasAndMapPoints(SceneMap *optimizedMap, bool eraseOutliers, int currentFrameID)
{
    // We optimize on all Cameras { Matrix3f<3, 4> } and all MapPoints { Vector3f }
    auto frames = optimizedMap->getKeyFrames();
    auto points = optimizedMap->getMapPoints();

    for (int i = 0; i < m_nIterations; ++i)
    {

        std::vector<Matrix<double, 3, 4>> optimizedExtr = std::vector<Matrix<double, 3, 4>>(frames.size());
        std::map<std::shared_ptr<MapPoint>, Vector3d> optimizedPoints;

        // Prepare BA constraints.
        ceres::Problem problem;
        prepareConstraints(frames, points, problem, optimizedExtr, optimizedPoints);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        //cout << "Global BA with " << optimizedExtr.size() << " poses and " << optimizedPoints.size() << " map points." << endl;

        //std::cout << "BA iteration " << i << " of " << m_nIterations << std::endl;
        clock_t begin = clock();
        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.BriefReport() << std::endl;
        // std::cout << summary.FullReport() << std::endl;

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        //std::cout << "BAOptimization iteration completed in " << elapsedSecs << " seconds." << std::endl;
    

        // Store result from optimizedPoses and optimizedPoints to the optimized SceneMap
        // for all mappoints from the original scenemap update thir position
        for (int i = 0; i < points.size(); ++i)
        {
            auto mappoint = points[i];

            Vector3f optPointPos = optimizedPoints.find(mappoint)->second.cast<float>();
            mappoint->setPosition(optPointPos);
        }
        // for all frames/cameras update their pose (assuming the same indexes in m_keyframes and optimizedPoses)
        for (int i = 0; i < frames.size(); ++i)
        {
            Matrix4f newExtr = Matrix4f::Identity();
            Matrix3f rot = optimizedExtr[i].block(0, 0, 3, 3).cast<float>();

            // make sure rotation is orthonormalized (https://math.stackexchange.com/questions/3292034/normalizing-a-quasi-rotation-matrix)
            Matrix3f a = ((rot.transpose() - rot) / (1. + rot.trace()));
            Matrix3f rotNormal = (Matrix3f::Identity() + a).inverse() * (Matrix3f::Identity() - a);

            newExtr.block(0, 0, 3, 3) = rotNormal;
            newExtr.block(0, 3, 3, 1) = optimizedExtr[i].block(0, 3, 3, 1).cast<float>();
            frames.at(i)->setPose(newExtr.inverse());

            // remove outlier from all frames
            pruneCorrespondences(frames.at(i), false);
        }
    }

    if (eraseOutliers)
    {
        SfMHelper::getInstance()->eraseOutlier(optimizedMap, currentFrameID);
    }
}

void GlobalBAOptimizer::prepareConstraints(const std::vector<std::shared_ptr<Frame>> &frames, const std::vector<std::shared_ptr<MapPoint>> &points, ceres::Problem &problem, std::vector<Matrix<double, 3, 4>> &optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d> &optimizedPoints) const
{

    for (int i = 0; i < points.size(); ++i)
    {
        auto point = points[i];

        Vector3d position = Vector3d(point->getPosition().cast<double>());

        optimizedPoints.insert(std::make_pair(point, position));
    }

    for (int i = 0; i < frames.size(); ++i)
    {
        auto frame = frames[i];
        auto visibleMapPoints = frame->getMapPoints();

        optimizedExtr[i] = frame->getPose().inverse().cast<double>().block(0, 0, 3, 4);

        for (int j = 0; j < visibleMapPoints.size(); ++j)
        {
            auto mapPoint = visibleMapPoints.at(j);

            // only process mappoints that were properly associated with keypoints and skip outliers
            if (mapPoint != nullptr && !frame->isOutlier(j))
            {

                Vector2f keypointPos = mapPoint->getCorresponding2DKeyPointPosition(frame);

                // loss function (soft selection)
                auto loss_function = new ceres::HuberLoss(sqrt(5.991));

                //don't optimize pose for keyframe 0 since it is our world anchor
                if (frame->getID() == 0) {
                    ceres::CostFunction* cost_function = PointOnlyReprojectionError::Create(keypointPos, frame->getPose().inverse(), frame->getIntrinsics());

                    problem.AddResidualBlock(cost_function,
                        loss_function,
                        (optimizedPoints.find(mapPoint)->second).data());
                }
                else {
                    ceres::CostFunction* cost_function = ReprojectionError::Create(keypointPos, frame->getIntrinsics());;

                    problem.AddResidualBlock(cost_function,
                        loss_function,
                        optimizedExtr[i].data(),
                        (optimizedPoints.find(mapPoint)->second).data());
                }
            }
        }
    }
}

// GlobalBAOptimizerAngles with optimizing angles for the frame/camera-pose - using Ceres for optimization.
void GlobalBAOptimizerAngles::optimizeCamerasAndMapPoints(SceneMap *optimizedMap, bool eraseOutliers, int currentFrameID)
{
    // We optimize on all Cameras {rotation[3], translation[3]} and MapPoints {Vector3f}
    auto frames = optimizedMap->getKeyFrames();
    auto points = optimizedMap->getMapPoints();

    for (int i = 0; i < m_nIterations; ++i)
    {

        std::vector<Matrix<double, 6, 1>> optimizedExtr = std::vector<Matrix<double, 6, 1>>(frames.size());
        std::map<std::shared_ptr<MapPoint>, Vector3d> optimizedPoints;

        // Prepare BA constraints.
        ceres::Problem problem;
        prepareConstraints(frames, points, problem, optimizedExtr, optimizedPoints);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        //std::cout << "Global BA with " << optimizedExtr.size() << " poses and " << optimizedPoints.size() << " map points." << endl;

        //std::cout << "BA iteration " << i << " of " << m_nIterations << std::endl;
        clock_t begin = clock();
        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.BriefReport() << std::endl;
        // std::cout << summary.FullReport() << std::endl;

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        //std::cout << "BAOptimization iteration completed in " << elapsedSecs << " seconds." << std::endl;

        // Store result from optimizedPoses and optimizedPoints to the optimized SceneMap
        // for all mappoints from the original scenemap update thir position
        for (int i = 0; i < points.size(); ++i)
        {
            auto mappoint = points[i];

            Vector3f optPointPos = optimizedPoints.find(mappoint)->second.cast<float>();
            mappoint->setPosition(optPointPos);
        }
        // for all frames/cameras update their pose (assuming the same indexes in m_keyframes and optimizedPoses)
        for (int i = 0; i < frames.size(); ++i)
        {
            Matrix4f newExtr = Matrix4f::Identity();
            Matrix<double, 3, 3> newRot;
            ceres::AngleAxisToRotationMatrix(optimizedExtr[i].block(0, 0, 3, 1).data(), newRot.data());
            newExtr.block(0, 0, 3, 3) = newRot.cast<float>();
            newExtr.block(0, 3, 3, 1) = optimizedExtr[i].block(3, 0, 3, 1).cast<float>();
            frames.at(i)->setPose(newExtr.inverse());

            // remove outlier from all frames
            pruneCorrespondences(frames.at(i), false);
        }
    }

    if (eraseOutliers)
    {
        SfMHelper::getInstance()->eraseOutlier(optimizedMap, currentFrameID);
    }
}
void GlobalBAOptimizerAngles::prepareConstraints(const std::vector<std::shared_ptr<Frame>> &frames, const std::vector<std::shared_ptr<MapPoint>> &points, ceres::Problem &problem, std::vector<Matrix<double, 6, 1>> &optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d> &optimizedPoints) const
{

    for (int i = 0; i < points.size(); ++i)
    {
        auto point = points[i];

        Vector3d position = Vector3d(point->getPosition().cast<double>());

        optimizedPoints.insert(std::make_pair(point, position));
    }

    for (int i = 0; i < frames.size(); ++i)
    {
        auto frame = frames[i];
        auto visibleMapPoints = frame->getMapPoints();

        Matrix4f extr = frame->getPose().inverse();
        Matrix<double, 3, 3> rotmd = extr.block(0, 0, 3, 3).cast<double>();
        ceres::RotationMatrixToAngleAxis(rotmd.data(), optimizedExtr[i].data());
        optimizedExtr[i].block(3, 0, 3, 1) = extr.block(0, 3, 3, 1).cast<double>();

        for (int j = 0; j < visibleMapPoints.size(); ++j)
        {
            auto mapPoint = visibleMapPoints.at(j);

            // only process mappoints that were properly associated with keypoints and skip outliers
            if (mapPoint != nullptr && !frame->isOutlier(j))
            {

                Vector2f keypointPos = mapPoint->getCorresponding2DKeyPointPosition(frame);
                
                // loss function (soft selection)
                auto loss_function = new ceres::HuberLoss(sqrt(5.991));

                //don't optimize pose for keyframe 0 since it is our world anchor
                if (frame->getID() == 0) {
                    ceres::CostFunction* cost_function = PointOnlyReprojectionError::Create(keypointPos, extr, frame->getIntrinsics());

                    problem.AddResidualBlock(cost_function,
                        loss_function,
                        (optimizedPoints.find(mapPoint)->second).data());
                }
                else {
                    ceres::CostFunction*  cost_function = AngleReprojectionError::Create(keypointPos, frame->getIntrinsics());

                    problem.AddResidualBlock(cost_function,
                        loss_function,
                        optimizedExtr[i].data(),
                        (optimizedPoints.find(mapPoint)->second).data());
                }
            }
        }
    }
}

// MotionOnlyBAOptimizer that optimizes only the camera poses and the mappoints are fixed
void MotionOnlyBAOptimizer::optimizeCameraPose(std::shared_ptr<Frame>frame)
{

    auto points = frame->getMapPoints();

    for (int i = 0; i < m_nIterations; ++i)
    {

        Matrix<double, 3, 4> optimizedExtr;
        std::map<std::shared_ptr<MapPoint>, Vector3d> optimizedPoints;

        // Prepare BA constraints

        ceres::Problem problem;
        // prepareConstraints(frame, points, m_nIterations<=1, problem, optimizedExtr, optimizedPoints);
        prepareConstraints(frame, points, true, problem, optimizedExtr, optimizedPoints);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        //cout << "Motion only BA with " << problem.NumResidualBlocks() << " map points." << endl;

        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.BriefReport() << std::endl;
        // std::cout << summary.FullReport() << std::endl;

        // Update the frame position with the optimized result
        Matrix4f newPose = Matrix4f::Identity();

        Matrix3f rot = optimizedExtr.block(0, 0, 3, 3).cast<float>();

        // make sure rotation is orthonormalized (https://math.stackexchange.com/questions/3292034/normalizing-a-quasi-rotation-matrix)
        Matrix3f a = ((rot.transpose() - rot) / (1. + rot.trace()));
        Matrix3f rotNormal = (Matrix3f::Identity() + a).inverse() * (Matrix3f::Identity() - a);

        newPose.block(0, 0, 3, 3) = rotNormal;
        newPose.block(0, 3, 3, 1) = optimizedExtr.block(0, 3, 3, 1).cast<float>();
        frame->setPose(newPose.inverse());

        pruneCorrespondences(frame);
    }
}

void MotionOnlyBAOptimizer::prepareConstraints(std::shared_ptr<Frame>frame, const std::vector<std::shared_ptr<MapPoint>> &points, bool useHuberLoss, ceres::Problem &problem, Matrix<double, 3, 4> &optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d> &optimizedPoints) const
{

    optimizedExtr = frame->getPose().inverse().cast<double>().block(0, 0, 3, 4);

    for (int j = 0; j < points.size(); ++j)
    {
        auto mapPoint = points.at(j);

        // only process mappoints that were properly associated with keypoints and skip outliers
        if (mapPoint != nullptr && !frame->isOutlier(j))
        {

            Vector3d pointPos = Vector3d(mapPoint->getPosition().cast<double>());
            optimizedPoints.insert(std::make_pair(mapPoint, pointPos));

            Vector2f keypointPos = mapPoint->getCorresponding2DKeyPointPosition(frame);

            ceres::CostFunction *cost_function = PoseOnlyReprojectionError::Create(keypointPos, frame->getIntrinsics(), mapPoint->getPosition().homogeneous());

            if (useHuberLoss)
            {
                // loss function (soft selection)
                auto loss_function = new ceres::HuberLoss(sqrt(5.991));
                problem.AddResidualBlock(cost_function, loss_function, optimizedExtr.data());
            }
            else
            {
                problem.AddResidualBlock(cost_function, nullptr, optimizedExtr.data());
            }
        }
    }
}

// MotionOnlyBAOptimizerAngles that optimizes only the camera poses with angles and the mappoints are fixed
void MotionOnlyBAOptimizerAngles::optimizeCameraPose(std::shared_ptr<Frame>frame)
{

    auto points = frame->getMapPoints();

    for (int i = 0; i < m_nIterations; ++i)
    {

        Matrix<double, 6, 1> optimizedExtr;
        std::map<std::shared_ptr<MapPoint>, Vector3d> optimizedPoints;

        // Prepare BA constraints

        ceres::Problem problem;
        // prepareConstraints(frame, points, m_nIterations <= 1, problem, optimizedExtr, optimizedPoints);
        prepareConstraints(frame, points, true, problem, optimizedExtr, optimizedPoints);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        //cout << "Motion only BA with " << problem.NumResidualBlocks() << " map points." << endl;

        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.BriefReport() << std::endl;
        // std::cout << summary.FullReport() << std::endl;

        // Update the frame position with the optimized result

        Matrix4f newExtr = Matrix4f::Identity();
        Matrix<double, 3, 3> newRot;
        ceres::AngleAxisToRotationMatrix(optimizedExtr.block(0, 0, 3, 1).data(), newRot.data());
        newExtr.block(0, 0, 3, 3) = newRot.cast<float>();
        newExtr.block(0, 3, 3, 1) = optimizedExtr.block(3, 0, 3, 1).cast<float>();
        frame->setPose(newExtr.inverse());

        pruneCorrespondences(frame);
    }
}

void MotionOnlyBAOptimizerAngles::prepareConstraints(std::shared_ptr<Frame>frame, const std::vector<std::shared_ptr<MapPoint>> &points, bool useHuberLoss, ceres::Problem &problem, Matrix<double, 6, 1> &optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d> &optimizedPoints) const
{

    Matrix4f pose = frame->getPose().inverse();
    Matrix<double, 3, 3> rotmd = pose.block(0, 0, 3, 3).cast<double>();
    ceres::RotationMatrixToAngleAxis(rotmd.data(), optimizedExtr.data());
    optimizedExtr.block(3, 0, 3, 1) = pose.block(0, 3, 3, 1).cast<double>();

    for (int j = 0; j < points.size(); ++j)
    {
        auto mapPoint = points.at(j);

        // only process mappoints that were properly associated with keypoints and skip outliers
        if (mapPoint != nullptr && !frame->isOutlier(j))
        {

            Vector3d pointPos = Vector3d(mapPoint->getPosition().cast<double>());
            optimizedPoints.insert(std::make_pair(mapPoint, pointPos));

            Vector2f keypointPos = mapPoint->getCorresponding2DKeyPointPosition(frame);

            ceres::CostFunction *cost_function =
                PoseOnlyAngleReprojectionError::Create(
                    keypointPos,
                    frame->getIntrinsics(),
                    mapPoint->getPosition().homogeneous());

            if (useHuberLoss)
            {
                // loss function (soft selection)
                auto loss_function = new ceres::HuberLoss(sqrt(5.991));
                problem.AddResidualBlock(cost_function, loss_function, optimizedExtr.data());
            }
            else
            {
                problem.AddResidualBlock(cost_function, nullptr, optimizedExtr.data());
            }
        }
    }
}

void LocalBAOptimizerAngles::optimizeCamerasAndMapPoints(SceneMap* optimizedMap, std::shared_ptr<Frame> currentFrame, bool eraseOutliers)
{
    //get ten best covisible frames of current frame
    std::vector<std::shared_ptr<Frame>> localFrames;
    std::vector<std::shared_ptr<MapPoint>> localMapPoints;
    std::vector<std::shared_ptr<Frame>> fixedFrames;

    // get best covisibility frames
    std::vector<std::shared_ptr<Frame>> covisibleFrames = currentFrame->getBestCovisibilityFrames(10);
    localFrames.insert(localFrames.begin(), covisibleFrames.begin(), covisibleFrames.end());
    localFrames.push_back(currentFrame);

    //accumulate distinct local mapPoints
    for (int i = 0; i < localFrames.size(); i++)
    {
        std::vector<std::shared_ptr<MapPoint>> cMapPoints = localFrames[i]->getMapPoints();

        for (int j = 0; j < cMapPoints.size(); j++)
        {
            std::shared_ptr<MapPoint> localMapPoint = cMapPoints[j];

            if (localMapPoint && !localFrames[i]->isOutlier(j) && !localMapPoint->isInvalid()) {
                if (std::find(localMapPoints.begin(), localMapPoints.end(), localMapPoint) == localMapPoints.end())
                {
                    // insert to
                    localMapPoints.push_back(localMapPoint);
                }
            }
        }
    }

    //additionally add frames observing the distinct mapPoints but are not optimized to add more constraint
    for (int i = 0; i < localMapPoints.size(); i++)
    {
        std::map<std::shared_ptr<Frame>, size_t> observingFrames = localMapPoints[i]->getObservingKeyframes();

        for (auto it = observingFrames.begin(); it != observingFrames.end(); it++)
        {
            if (!it->first->isOutlier(it->second) && it->first->isKeyFrame()) {
                if (std::find(localFrames.begin(), localFrames.end(), it->first) == localFrames.end()) {
                    if (std::find(fixedFrames.begin(), fixedFrames.end(), it->first) == fixedFrames.end()) {
                        fixedFrames.push_back(it->first);
                    }
                }
            }
        }
    }

    //cout << "#Frames: " << localFrames.size() + fixedFrames.size() << " of " << optimizedMap->getKeyFrameCount() << " #Points " << localMapPoints.size() << " of " << optimizedMap->getMapPointCount() << endl;

    for (int i = 0; i < m_nIterations; ++i)
    {

        std::vector<Matrix<double, 6, 1>> optimizedExtr = std::vector<Matrix<double, 6, 1>>(localFrames.size());
        std::map<std::shared_ptr<MapPoint>, Vector3d> optimizedPoints;

        // Prepare BA constraints.
        ceres::Problem problem;
        prepareConstraints(localFrames, fixedFrames, localMapPoints, problem, optimizedExtr, optimizedPoints);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        //std::cout << "Global BA with " << optimizedExtr.size() << " poses and " << optimizedPoints.size() << " map points." << endl;

        //std::cout << "BA iteration " << i << " of " << m_nIterations << std::endl;
        clock_t begin = clock();
        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.BriefReport() << std::endl;
        // std::cout << summary.FullReport() << std::endl;

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        //std::cout << "BAOptimization iteration completed in " << elapsedSecs << " seconds." << std::endl;

        // Store result from optimizedPoses and optimizedPoints to the optimized SceneMap
        // for all mappoints from the original scenemap update thir position
        for (int i = 0; i < localMapPoints.size(); ++i)
        {
            auto mappoint = localMapPoints[i];

            Vector3f optPointPos = optimizedPoints.find(mappoint)->second.cast<float>();
            mappoint->setPosition(optPointPos);
        }
        // for all frames/cameras update their pose (assuming the same indexes in m_keyframes and optimizedPoses)
        for (int i = 0; i < localFrames.size(); ++i)
        {
            Matrix4f newExtr = Matrix4f::Identity();
            Matrix<double, 3, 3> newRot;
            ceres::AngleAxisToRotationMatrix(optimizedExtr[i].block(0, 0, 3, 1).data(), newRot.data());
            newExtr.block(0, 0, 3, 3) = newRot.cast<float>();
            newExtr.block(0, 3, 3, 1) = optimizedExtr[i].block(3, 0, 3, 1).cast<float>();
            localFrames.at(i)->setPose(newExtr.inverse());

            // remove outlier from all frames
            pruneCorrespondences(localFrames.at(i), false);
        }

        for (int i = 0; i < fixedFrames.size(); i++) {
            pruneCorrespondences(fixedFrames[i], false);
        }
    }

    if (eraseOutliers)
    {
        SfMHelper::getInstance()->eraseOutlier(optimizedMap, currentFrame->getID());
    }
}

void LocalBAOptimizerAngles::prepareConstraints(const std::vector<std::shared_ptr<Frame>>& localFrames, const std::vector<std::shared_ptr<Frame>>& fixedFrames, const std::vector<std::shared_ptr<MapPoint>>& points, ceres::Problem& problem, std::vector<Matrix<double, 6, 1>>& optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d>& optimizedPoints) const
{

    for (int i = 0; i < points.size(); ++i)
    {
        auto point = points[i];

        Vector3d position = Vector3d(point->getPosition().cast<double>());

        optimizedPoints.insert(std::make_pair(point, position));
    }

    for (int i = 0; i < localFrames.size(); ++i)
    {
        auto frame = localFrames[i];
        auto visibleMapPoints = frame->getMapPoints();

        Matrix4f extr = frame->getPose().inverse();
        Matrix<double, 3, 3> rotmd = extr.block(0, 0, 3, 3).cast<double>();
        ceres::RotationMatrixToAngleAxis(rotmd.data(), optimizedExtr[i].data());
        optimizedExtr[i].block(3, 0, 3, 1) = extr.block(0, 3, 3, 1).cast<double>();

        for (int j = 0; j < visibleMapPoints.size(); ++j)
        {
            auto mapPoint = visibleMapPoints.at(j);

            // only process mappoints that were properly associated with keypoints and skip outliers
            if (mapPoint != nullptr && !frame->isOutlier(j))
            {

                Vector2f keypointPos = mapPoint->getCorresponding2DKeyPointPosition(frame);

                // loss function (soft selection)
                auto loss_function = new ceres::HuberLoss(sqrt(5.991));

                //don't optimize pose for keyframe 0 since it is our world anchor
                if (frame->getID() == 0) {
                    ceres::CostFunction* cost_function = PointOnlyReprojectionError::Create(keypointPos, extr, frame->getIntrinsics());

                    problem.AddResidualBlock(cost_function,
                        loss_function,
                        (optimizedPoints.find(mapPoint)->second).data());
                }
                else {
                    ceres::CostFunction* cost_function = AngleReprojectionError::Create(keypointPos, frame->getIntrinsics());

                    problem.AddResidualBlock(cost_function,
                        loss_function,
                        optimizedExtr[i].data(),
                        (optimizedPoints.find(mapPoint)->second).data());
                }
            }
        }
    }

    
    for (int i = 0; i < fixedFrames.size(); ++i)
    {
        auto frame = fixedFrames[i];
        auto visibleMapPoints = frame->getMapPoints();

        for (int j = 0; j < visibleMapPoints.size(); ++j)
        {
            auto mapPoint = visibleMapPoints.at(j);

            // only process mappoints that were properly associated with keypoints and skip outliers
            if (mapPoint != nullptr && !frame->isOutlier(j))
            {
                //only add constraint for point that is existing in list
                if (optimizedPoints.find(mapPoint) != optimizedPoints.end()) {

                    Vector2f keypointPos = mapPoint->getCorresponding2DKeyPointPosition(frame);

                    // loss function (soft selection)
                    auto loss_function = new ceres::HuberLoss(sqrt(5.991));

                    //don't optimize pose for keyframe 0 since it is our world anchor
                    ceres::CostFunction* cost_function = PointOnlyReprojectionError::Create(keypointPos, frame->getPose().inverse(), frame->getIntrinsics());

                    problem.AddResidualBlock(cost_function,loss_function,(optimizedPoints.find(mapPoint)->second).data());
                }
            }
        }

    }

}
