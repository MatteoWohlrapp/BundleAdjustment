#include "SfMHelper.h"

using namespace cv;


SfMHelper* SfMHelper::inst_ = NULL;

SfMHelper* SfMHelper::getInstance()
{
    if (inst_ == NULL) {
        inst_ = new SfMHelper();
    }
    return (inst_);
}

void SfMHelper::estimatePoseUsingPnP(std::shared_ptr<Frame> frame)
{
    // convert data to opencv
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    // loop through all map points in frame by index, add them to object points and image points if not an outlier, otherwise skip, make sure to adjust index accordingly
    std::map<int, int> pnp_to_frame_points;
    // get map points from frame

    int pnp_index = 0;
    for (int i = 0; i < frame->getKeypointCount(); i++)
    {
        if (!frame->isOutlier(i) && frame->getMapPoint(i))
        {
            pnp_to_frame_points[pnp_index] = i;
            pnp_index++;
            Eigen::Vector3f objectPointEigen = frame->getMapPoint(i)->getPosition();
            cv::Point3f objectPointCV(objectPointEigen.x(), objectPointEigen.y(), objectPointEigen.z());
            objectPoints.push_back(objectPointCV);

            Eigen::Vector2f imagePointEigen = frame->getMapPoint(i)->getCorresponding2DKeyPointPosition(frame);
            cv::Point2f imagePointCV(imagePointEigen.x(), imagePointEigen.y());
            imagePoints.push_back(imagePointCV);
        }
    }

    //std::cout << "Image Points: " << imagePoints.size() << "; Object Points: " << objectPoints.size() << " objectPoints." << std::endl;

    //if no points are inliers or we have fewer than 6 2d-3d correspondences, just keep constant speed assumption
    if (objectPoints.empty() || imagePoints.empty() || imagePoints.size() <= 6)
    {
        return;
    }

    cv::Mat cvintrinsics;
    cv::eigen2cv(frame->getIntrinsics(), cvintrinsics);

    // SolvePnP
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);
    cv::Mat rvec;
    cv::Mat tvec = cv::Mat(3, 1, CV_32F);

    // initialize rvec, tvec with intitial pose estimate
    Matrix4f initialExtr = frame->getPose().inverse();
    Eigen::Matrix3f rot = initialExtr.block(0, 0, 3, 3);
    Vector3f trans = initialExtr.block(0, 3, 3, 1);
    tvec.at<float>(0) = trans.x();
    tvec.at<float>(1) = trans.y();
    tvec.at<float>(2) = trans.z();

    cv::Mat rotMat;
    cv::eigen2cv(rot, rotMat);
    cv::Rodrigues(rotMat, rvec);

    std::vector<int> inliers;
    //bool success = cv::solvePnPRansac(objectPoints, imagePoints, cvintrinsics, distCoeffs, rvec, tvec, true, 1000, 8.0, 0.99, inliers);//, SOLVEPNP_ITERATIVE);
    bool success = cv::solvePnP(objectPoints, imagePoints, cvintrinsics, distCoeffs, rvec, tvec, true);

    //due to a bug with solvepnp, we sometimes observe huge normes. Avoid this by just keeping constant speed assumption
    float norm = cv::norm(tvec, cv::NORM_L2);
    if (norm >= 1.)
    {
        return;
    }

    if (success)
    {
        int inlierCount = 0;
        for (int i = 0; i < inliers.size(); i++)
        {
            if (inliers[i] == 0)
            {
                // convert index from pnp to frame
                int frame_index = pnp_to_frame_points[i];
                frame->setOutlier(frame_index);
            }
            else {
                inlierCount++;
            }
        }
        cv::Mat rotationMatrix;
        cv::Rodrigues(rvec, rotationMatrix);

        Matrix4f eigenMatrix = convertToMatrix4f(rotationMatrix, tvec);

        frame->setPose(eigenMatrix.inverse());
    }
}

void SfMHelper::estimatePoseUsingBA(std::shared_ptr<Frame>frame)
{
    //cout << "Initial Pose \n" << frame->getPose() << endl;
    
    // optimize pose with motion only BA
    MotionOnlyBAOptimizerAngles optimizer = MotionOnlyBAOptimizerAngles();
    optimizer.setNbOfIterations(4);
    optimizer.setNbOfMaxItPerBA(20);
    optimizer.optimizeCameraPose(frame);

    //cout << "Optimized Pose \n" << frame->getPose() << endl;
}

void SfMHelper::setPoseEstimationType(EstimationType type)
{
    estimation = type;
}

void SfMHelper::estimatePose(std::shared_ptr<Frame>frame, std::shared_ptr<Frame> lastFrame, std::vector<cv::DMatch>* matches)
{
    switch (estimation)
    {
    case PNP:
    {   
        estimatePoseUsingPnP(frame);
    }
    break;
    case BA:
    {
        estimatePoseUsingBA(frame);
    }
    break;
    case EssentialOrHomography:
    {
        Matrix4f pose = Matrix4f::Identity();
        if (recoverPose(lastFrame, frame, matches, pose)) {
            frame->setPose(lastFrame->getPose() * pose);
        }
        else {
            //For now just keep constant speed in case it fails
        }
        
    }
    break;
    default:
        break;
    }
}

Matrix4f SfMHelper::setUpDebugCam(std::vector<cv::Point3f> p3d, std::vector<Point2f> &points1, std::vector<Point2f> &points2, cv::Mat &intr)
{
    points1.clear();
    points2.clear();

    float f = 100, w = 640, h = 480;

    intr = cv::Mat(3, 3, CV_64FC1);
    intr.at<double>(0, 0) = f;
    intr.at<double>(0, 1) = 0.0;
    intr.at<double>(0, 2) = w / 2;
    intr.at<double>(1, 0) = 0.0;
    intr.at<double>(1, 1) = f;
    intr.at<double>(1, 2) = h / 2;
    intr.at<double>(2, 0) = 0.;
    intr.at<double>(2, 1) = 0.;
    intr.at<double>(2, 2) = 1;

    // set transformation from 1st to 2nd camera (assume K is unchanged)
    cv::Mat1f rvecDeg = (cv::Mat1f(3, 1) << 45, 0, 0);
    cv::Mat1f t = (cv::Mat1f(3, 1) << 0, 0, 1);

    cv::Mat rMat;
    cv::Rodrigues(rvecDeg * CV_PI / 180, rMat);
    // std::cout << "-------------------------------------------\n";
    // std::cout << "Ground truth:\n";

    // std::cout << "K = \n" << intr << std::endl << std::endl;
    Matrix4f extr = convertToMatrix4f(rMat, t);
    // cout << extr.inverse() << endl;

    // project on both cameras
    cv::projectPoints(p3d,
                      cv::Mat1d::zeros(3, 1),
                      cv::Mat1d::zeros(3, 1),
                      intr,
                      cv::Mat(),
                      points1);

    cv::projectPoints(p3d,
                      rvecDeg * CV_PI / 180,
                      t,
                      intr,
                      cv::Mat(),
                      points2);
    return extr;
}

void SfMHelper::searchInNeighbors(std::shared_ptr<Frame>frame, FeatureProcessor *fp, SceneMap *map)
{
    float chiThresh = 5.991;
    float sigmaFactor = 1.2;
    std::vector<std::shared_ptr<Frame>> allNeighbors;
    // get best covisibility frames
    std::vector<std::shared_ptr<Frame>> covisibleFrames = frame->getBestCovisibilityFrames(20);
    allNeighbors.insert(allNeighbors.begin(), covisibleFrames.begin(), covisibleFrames.end());

    // also look in neighbors of best covisible frames
    for (int i = 0; i < covisibleFrames.size(); i++)
    {

        std::vector<std::shared_ptr<Frame>> ccFrames = covisibleFrames[i]->getBestCovisibilityFrames(5);

        for (int j = 0; j < ccFrames.size(); j++)
        {
            std::shared_ptr<Frame>covisibleFrame = ccFrames[j];

            if (covisibleFrame == frame)
                continue;

            if (std::find(allNeighbors.begin(), allNeighbors.end(), covisibleFrame) == allNeighbors.end())
            {
                // insert to
                allNeighbors.push_back(covisibleFrame);
            }
        }
    }

    int triaCount = 0;
    int fusedCount = 0;
    int transferredToFrame = 0;
    int transferredToNeighbor = 0;
    for (int i = 0; i < allNeighbors.size(); i++)
    {
        std::shared_ptr<Frame>neighbor = allNeighbors[i];

        // compute matches
        std::vector<cv::DMatch> matches = fp->matchFeatures(frame, neighbor);

        std::vector<DMatch> triangulateMatches;

        // process matches
        for (int j = 0; j < matches.size(); j++)
        {
            DMatch match = matches[j];

            int indFrame = match.queryIdx;
            int indNeighbor = match.trainIdx;

            if (!frame->isOutlier(indFrame) && !neighbor->isOutlier(indNeighbor))
            {

                std::shared_ptr<MapPoint>frameMP = frame->getMapPoint(indFrame);
                std::shared_ptr<MapPoint>neighborMP = neighbor->getMapPoint(indNeighbor);
                // no points associated => triangulate new point
                if (!frameMP && !neighborMP)
                {

                    triangulateMatches.push_back(match);
                    triaCount++;
                }
                else if (!frameMP && neighborMP)
                {

                    // check if valid connection
                    KeyPoint *keypoint = frame->getKeypoint(indFrame);

                    Vector3f camCoord = (frame->getPose().inverse() * neighborMP->getPosition().homogeneous()).hnormalized();

                    // check z positive
                    if (camCoord.z() <= 0.)
                    {
                        continue;
                    }

                    Vector2f pixel = (frame->getIntrinsics() * camCoord).hnormalized();

                    // check image bounds
                    if (pixel.x() < 0 || pixel.x() >= frame->getWidth() || pixel.y() < 0 || pixel.y() >= frame->getHeight())
                    {
                        continue;
                    }

                    float invSigmaP = 1.0 / pow(1.2, keypoint->octave);
                    Vector2f pointObs = Vector2f(keypoint->pt.x, keypoint->pt.y);
                    float squaredDist = (pixel - pointObs).norm();

                    float chiSquare = squaredDist * invSigmaP;

                    // check reprojection error
                    if (chiSquare > chiThresh)
                    {
                        continue;
                    }

                    // check viewing angle (we want smaller than 60 degrees)
                    Vector3f worldDir = (neighborMP->getPosition() - frame->getWorldPos()).normalized();
                    Vector3f viewingNormal = neighborMP->getViewingDirection();

                    if (worldDir.dot(viewingNormal) < 0.5)
                    {
                        continue;
                    }

                    // check if depth bounds mappoint is inside the scale pyramid of the points reference frame
                    float worldDist = (neighborMP->getPosition() - frame->getWorldPos()).norm();
                    float minDist = neighborMP->getMinDistance();
                    float maxDist = neighborMP->getMaxDistance();
                    if (worldDist > maxDist || worldDist < minDist)
                    {
                        continue;
                    }

                    // check if neighborMP is not already observed by frame
                    if (!neighborMP->isObservedBy(frame))
                    {
                        // neighborMP has associated map point => add reference to frameMP

                        // check descriptor distance of point
                        float distMapPoint = norm(frame->getDescriptor(match.queryIdx), neighborMP->getDescriptor(), NORM_L2);

                        if (distMapPoint < 0.2)
                        {
                            neighborMP->addObservation(frame, indFrame);
                            frame->addAssociatedMapPoint(indFrame, neighborMP);
                            transferredToFrame++;
                        }
                    }
                    else
                    {
                        // check descriptor distance to MP of both observations
                        int indObserved = neighborMP->getObservedIndex(frame);
                        cv::Mat descObserved = frame->getDescriptor(indObserved);
                        cv::Mat descMatched = frame->getDescriptor(indFrame);

                        float distObserved = norm(descObserved, neighborMP->getDescriptor(), NORM_L2);
                        float distMatched = norm(descMatched, neighborMP->getDescriptor(), NORM_L2);

                        // only update if new match is better than old
                        if (distMatched < distObserved)
                        {
                            // reassign match in keyframe
                            frame->eraseAssociatedMapPoint(indObserved);
                            frame->addAssociatedMapPoint(indFrame, neighborMP);

                            // reassign match in mapPoint
                            neighborMP->replaceObservation(frame, indFrame);
                            transferredToFrame++;
                        }
                    }
                }
                else if (!neighborMP && frameMP)
                {

                    // check if valid connection
                    KeyPoint *keypoint = neighbor->getKeypoint(indNeighbor);

                    Vector3f camCoord = (neighbor->getPose().inverse() * frameMP->getPosition().homogeneous()).hnormalized();

                    // check z positive
                    if (camCoord.z() <= 0.)
                    {
                        continue;
                    }

                    Vector2f pixel = (neighbor->getIntrinsics() * camCoord).hnormalized();

                    // check image bounds
                    if (pixel.x() < 0 || pixel.x() >= neighbor->getWidth() || pixel.y() < 0 || pixel.y() >= neighbor->getHeight())
                    {
                        continue;
                    }

                    float invSigmaP = 1.0 / pow(1.2, keypoint->octave);
                    Vector2f pointObs = Vector2f(keypoint->pt.x, keypoint->pt.y);
                    float squaredDist = (pixel - pointObs).norm();

                    float chiSquare = squaredDist * invSigmaP;

                    // check reprojection error
                    if (chiSquare > chiThresh)
                    {
                        continue;
                    }

                    // check viewing angle (we want smaller than 60 degrees)
                    Vector3f worldDir = (frameMP->getPosition() - neighbor->getWorldPos()).normalized();
                    Vector3f viewingNormal = frameMP->getViewingDirection();

                    if (worldDir.dot(viewingNormal) < 0.5)
                    {
                        continue;
                    }

                    // check if depth bounds mappoint is inside the scale pyramid of the points reference frame
                    float worldDist = (frameMP->getPosition() - neighbor->getWorldPos()).norm();
                    float minDist = frameMP->getMinDistance();
                    float maxDist = frameMP->getMaxDistance();
                    if (worldDist > maxDist || worldDist < minDist)
                    {
                        continue;
                    }

                    // check if frameMP is not already observed by neighbor
                    if (!frameMP->isObservedBy(neighbor))
                    {
                        // frameMP has associated map point => add reference to neighborMP

                        // check descriptor distance of point
                        float distMapPoint = norm(frameMP->getDescriptor(), neighbor->getDescriptor(match.trainIdx), NORM_L2);

                        if (distMapPoint < 0.2)
                        {
                            frameMP->addObservation(neighbor, indNeighbor);
                            neighbor->addAssociatedMapPoint(indNeighbor, frameMP);
                            transferredToNeighbor++;
                        }
                    }
                    else
                    {
                        // check descriptor distance to MP of both observations
                        int indObserved = frameMP->getObservedIndex(neighbor);
                        cv::Mat descObserved = neighbor->getDescriptor(indObserved);
                        cv::Mat descMatched = neighbor->getDescriptor(indNeighbor);

                        float distObserved = norm(descObserved, frameMP->getDescriptor(), NORM_L2);
                        float distMatched = norm(descMatched, frameMP->getDescriptor(), NORM_L2);

                        // only update if new match is better than old
                        if (distMatched < distObserved)
                        {
                            // reassign match in keyframe
                            neighbor->eraseAssociatedMapPoint(indObserved);
                            neighbor->addAssociatedMapPoint(indNeighbor, frameMP);

                            // reassign match in mapPoint
                            frameMP->replaceObservation(neighbor, indNeighbor);
                            transferredToNeighbor++;
                        }
                    }
                }
                else
                {
                    // both points associated => fuse map points: keep point with more observations and transfer observations to other
                    // Check viewing direction compatible
                    Vector3f viewDirFrame = frameMP->getViewingDirection();
                    Vector3f viewDirNeighbor = neighborMP->getViewingDirection();

                    if (viewDirFrame.dot(viewDirNeighbor) < 0.0)
                    {
                        continue;
                    }

                    // check descriptor distance
                    float descDist = norm(frameMP->getDescriptor(), neighborMP->getDescriptor(), NORM_L2);
                    if (descDist < 0.3)
                    {
                        if (frameMP->getNumObserved() > neighborMP->getNumObserved())
                        {
                            if (neighborMP->fuse(frameMP))
                            {
                                map->eraseMapPoint(neighborMP);
                            }
                        }
                        else
                        {
                            if (frameMP->fuse(neighborMP))
                            {
                                map->eraseMapPoint(frameMP);
                            }
                        }

                        fusedCount++;
                    }
                }
            }
        }

        // triangulate points
        if (triangulateMatches.size() > 0)
        {
            triangulatePoints(frame, neighbor, &triangulateMatches, map, true);
        }
    }

    //cout << "#Triangulated: " << triaCount << endl;
    //cout << "#Fused: " << fusedCount << endl;
    //cout << "#Transferred to frame: " << transferredToFrame << endl;
    //cout << "#Transferred to neighbor: " << transferredToNeighbor << endl;

    frame->updateCovisibilityGraph();
}

bool SfMHelper::recoverPose(std::shared_ptr<Frame>frame1, std::shared_ptr<Frame>frame2, std::vector<DMatch> *matches, Matrix4f &pose)
{

    // init values for scoring (similar to the way orb slam does it)
    double sigma = 1.0;
    const double th = 3.841;
    const double thScore = 5.991;
    const double invSigmaSquare = 1.0 / (sigma * sigma);

    // Extract matched keypoints from both frames
    std::vector<KeyPoint> keypoints1 = frame1->getKeypoints();
    std::vector<KeyPoint> keypoints2 = frame2->getKeypoints();

    // Convert keypoints to Point2f
    std::vector<Point2f> points1, points2;

    for (int i = 0; i < matches->size(); i++)
    {
        DMatch match = (*matches)[i];
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat intrinsicMatrix;
    cv::eigen2cv(frame1->getIntrinsics(), intrinsicMatrix);
    intrinsicMatrix.convertTo(intrinsicMatrix, CV_64FC1);

    // std::vector<cv::Point3f> p3d{ {0, 0, 10},{1, 0, 10},{2, 0, 10}, {100, 0, 10},{0, 1, 10},{0, 2, 10},{0, 100, 10},{1, 1, 10},{2, 2, 10},{100, 100, 10} };
    // setUpDebugCam(p3d,points1, points2, intrinsicMatrix);

    // compute fundamental mat
    // mask indicates inliers/outliers for recovered pose
    cv::Mat maskH, maskE;
    // Calculate the Essential Matrix
    cv::Mat essentialMat = cv::findEssentialMat(points1, points2, intrinsicMatrix, RANSAC, 0.999, 1.0, maskE);

    // cv::Mat fundamentalMat = cv::findFundamentalMat(points2, points1, maskF, RANSAC,0.1,0.99);
    // compute homography
    cv::Mat homographyMat = cv::findHomography(points1, points2, RANSAC, 1.0, maskH, 2000, 0.999);

    // compute average reprojection error as heuristic
    std::vector<cv::Point2f> HReprojection12(points1.size());
    cv::perspectiveTransform(points1, HReprojection12, homographyMat);
    std::vector<cv::Point2f> HReprojection21(points2.size());
    cv::perspectiveTransform(points2, HReprojection21, homographyMat.inv());

    double homographyScore = 0.;

    int HInl = 0;
    for (size_t i = 0; i < points1.size(); i++)
    {
        // only consider inliers
        if (maskH.at<bool>(i, 0))
        {
            // squared distance
            double sqrDst1 = (points2[i] - HReprojection12[i]).ddot(points2[i] - HReprojection12[i]);
            double sqrDst2 = (points1[i] - HReprojection21[i]).ddot((points1[i] - HReprojection21[i]));

            double chiSquare1 = sqrDst1 * invSigmaSquare;
            double chiSquare2 = sqrDst2 * invSigmaSquare;

            if (chiSquare1 > thScore)
            {
                // mark as additional outlier
                maskH.at<bool>(i, 0) = false;
            }
            else
            {
                homographyScore += thScore - chiSquare1;
            }

            if (chiSquare2 > thScore)
            {
                // mark as additional outlier
                maskH.at<bool>(i, 0) = false;
            }
            else
            {
                homographyScore += thScore - chiSquare2;
            }

            // if still inlier count it
            if (maskH.at<bool>(i, 0))
            {
                HInl++;
            }
        }
    }

    //std::cout << "Homography Inlier: " << HInl << " Homography score: " << homographyScore << endl;

    cv::Mat fundamentalMat = intrinsicMatrix.inv().t() * essentialMat * intrinsicMatrix.inv();
    std::vector<cv::Point3f> FEpiline12(points1.size());
    cv::computeCorrespondEpilines(points1, 1, fundamentalMat, FEpiline12);
    std::vector<cv::Point3f> FEpiline21(points2.size());
    cv::computeCorrespondEpilines(points2, 2, fundamentalMat, FEpiline21);

    double fundamentalScore = 0.;
    int FInl = 0;
    for (size_t i = 0; i < points1.size(); i++)
    {
        // only consider inliers
        if (maskE.at<bool>(i, 0))
        {
            // point to line distance squared
            double numerator12 = (points2[i].x * FEpiline12[i].x) + (points2[i].y * FEpiline12[i].y) + FEpiline12[i].z;
            double sqrDst1 = numerator12 * numerator12 / ((FEpiline12[i].x * FEpiline12[i].x) + (FEpiline12[i].y * FEpiline12[i].y));
            double numerator21 = (points1[i].x * FEpiline21[i].x) + (points1[i].y * FEpiline21[i].y) + FEpiline21[i].z;
            double sqrDst2 = numerator21 * numerator21 / ((FEpiline21[i].x * FEpiline21[i].x) + (FEpiline21[i].y * FEpiline21[i].y));

            double chiSquare1 = sqrDst1 * invSigmaSquare;
            double chiSquare2 = sqrDst2 * invSigmaSquare;

            if (chiSquare1 > th)
            {
                // mark as additional outlier
                maskE.at<bool>(i, 0) = false;
            }
            else
            {
                fundamentalScore += thScore - chiSquare1;
            }

            if (chiSquare2 > th)
            {
                // mark as additional outlier
                maskE.at<bool>(i, 0) = false;
            }
            else
            {
                fundamentalScore += thScore - chiSquare2;
            }

            // if still inlier count it
            if (maskE.at<bool>(i, 0))
            {
                FInl++;
            }
        }
    }

    //std::cout << "Fundamental Inlier: " << FInl << " Fundamental score: " << fundamentalScore << endl;

    // select best result based on heuristic
    float ratio = homographyScore / (homographyScore + fundamentalScore);
    bool useH = ratio > 0.4;

    //std::cout << "Ratio " << ratio << endl;
    // decompose chosen to get rotation and translation
    if (!useH)
    {
        // Fundamental/Essential
        //std::cout << "Using Essential Mat" << endl;

        cv::Mat R, t;
        int inliers = cv::recoverPose(essentialMat, points1, points2, intrinsicMatrix, R, t, maskE);
        //std::cout << "#Inliers " << inliers << endl;

        if (inliers <= 100)
        {
            return false;
        }

        // Convert R, t to a 4x4 transformation matrix (Matrix4f)
        pose = convertToMatrix4f(R, t).inverse();

        // mark outlier points
        for (int i = 0; i < matches->size(); i++)
        {
            DMatch match = (*matches)[i];
            bool inlier = maskE.at<bool>(i, 0);
            if (!inlier)
            {
                frame1->setOutlier(match.queryIdx);
                frame2->setOutlier(match.trainIdx);
            }
        }

        return true;
    }
    else
    {
        // Homography
        //std::cout << "Using Homography Mat" << endl;

        std::vector<cv::Mat> Rs, ts, ns;
        decomposeHomographyMat(homographyMat, intrinsicMatrix, Rs, ts, ns);

        std::vector<int> res;
        // rectify points
        std::vector<Point2f> recPoints1, recPoints2;
        cv::undistortPoints(points1, recPoints1, intrinsicMatrix, cv::Mat());
        cv::undistortPoints(points2, recPoints2, intrinsicMatrix, cv::Mat());
        filterHomographyDecompByVisibleRefpoints(Rs, ns, recPoints1, recPoints2, res, maskH);

        if (res.size() > 1)
        {
            // we have to find out best solution

            // take solution with z value closest to 1 or -1, since our source plane is the projection plane of camera1 and the normal coordinate is given in camera1's local cordinates (OpenCV assumes positive z in front of camera in camera coords)
            double largestZ = fabs(ns[res[0]].at<double>(2, 0));
            int poseIndex = res[0];
            for (int i = 1; i < res.size(); i++)
            {
                int ind = res[i];

                double z = fabs(ns[ind].at<double>(2, 0));

                if (z > largestZ)
                {
                    largestZ = z;
                    poseIndex = ind;
                }
            }

            pose = convertToMatrix4f(Rs[poseIndex], ts[poseIndex]).inverse();
        }
        else if (res.size() == 1)
        {
            //std::cout << "Unique result" << endl;
            // Convert R, t to a 4x4 transformation matrix (Matrix4f)
            pose = convertToMatrix4f(Rs[res[0]], ts[res[0]]).inverse();
        }
        else
        {
            return false;
        }

        // mark outlier points
        for (int i = 0; i < matches->size(); i++)
        {
            DMatch match = (*matches)[i];
            bool inlier = maskH.at<bool>(i, 0);
            if (!inlier)
            {
                frame1->setOutlier(match.queryIdx);
                frame2->setOutlier(match.trainIdx);
            }
        }

        return true;
    }

    return false;
}

Eigen::Matrix4f SfMHelper::convertToMatrix4f(cv::Mat R, cv::Mat t)
{
    // Create an identity matrix
    Eigen::Matrix4f combined = Eigen::Matrix4f::Identity();

    Eigen::Matrix3f rot;
    Eigen::Vector3f trans;
    cv2eigen(R, rot);
    cv2eigen(t, trans);
    combined.block(0, 0, 3, 3) = rot;
    combined.block(0, 3, 3, 1) = trans;

    return combined;
}

void SfMHelper::triangulatePoints(std::shared_ptr<Frame>frame1, std::shared_ptr<Frame>frame2, std::vector<DMatch> *matches, SceneMap *map, bool checkBaseline)
{

    // check if baseline is big enough (compare distance between cameras with median map point depth in frame 2)
    if (checkBaseline)
    {
        float baseLine = (frame2->getWorldPos() - frame1->getWorldPos()).norm();
        float ratioBaseLine = baseLine / frame2->getMedianMapPointDepth();
        if (ratioBaseLine < 0.01)
        {
            return;
        }
    }

    // Projection matrices for both frames
    cv::Mat projMatrix1 = getCVProjectionMatrix(frame1);
    cv::Mat projMatrix2 = getCVProjectionMatrix(frame2);

    std::vector<KeyPoint> keypoints1 = frame1->getKeypoints();
    std::vector<KeyPoint> keypoints2 = frame2->getKeypoints();

    // Convert matched keypoints to homogeneous coordinates
    std::vector<Point2f> points1, points2;
    for (int i = 0; i < matches->size(); i++)
    {
        DMatch match = (*matches)[i];
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Triangulate points
    cv::Mat points4D;
    cv::triangulatePoints(projMatrix1, projMatrix2, points1, points2, points4D);

    int badCounter = 0;
    // Convert 4D points to 3D
    for (int i = 0; i < points4D.cols; ++i)
    {
        Vector3f newPoint;
        Point3f pt;
        pt.x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
        pt.y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
        pt.z = points4D.at<float>(2, i) / points4D.at<float>(3, i);
        newPoint = Vector3f(pt.x, pt.y, pt.z);

        // sanity check
        // check if in front of both cams
        Vector3f camCoord1 = (frame1->getPose().inverse() * newPoint.homogeneous()).hnormalized();
        Vector3f camCoord2 = (frame2->getPose().inverse() * newPoint.homogeneous()).hnormalized();

        if (camCoord1.z() <= 0. && camCoord2.z() <= 0.)
        {
            badCounter++;
            continue;
        }

        // check distance between projected triangulated and initial points
        float chiThresh = 5.991;
        float sigmaFactor = 1.2;
        float octaveScale1 = pow(1.2, frame1->getKeypoint((*matches)[i].queryIdx)->octave);
        float octaveScale2 = pow(1.2, frame2->getKeypoint((*matches)[i].trainIdx)->octave);
        float invSigmaP1 = 1.0 / octaveScale1;
        float invSigmaP2 = 1.0 / octaveScale2;

        Vector2f pixel1 = (frame1->getIntrinsics() * camCoord1).hnormalized();
        Vector2f pixel2 = (frame2->getIntrinsics() * camCoord2).hnormalized();
        Vector2f point1E = Vector2f(points1[i].x, points1[i].y);
        Vector2f point2E = Vector2f(points2[i].x, points2[i].y);

        float squaredDist1 = (pixel1 - point1E).norm();
        float squaredDist2 = (pixel2 - point2E).norm();

        float chiSquare1 = squaredDist1 * invSigmaP1;
        float chiSquare2 = squaredDist2 * invSigmaP2;

        if (chiSquare1 > chiThresh || chiSquare2 > chiThresh)
        {
            badCounter++;
            continue;
        }

        // Check scale consistency (assure that ratio of distances is roughly equal to inverse ratio of octave scales)
        float worldDist1 = (newPoint - frame1->getWorldPos()).norm();
        float worldDist2 = (newPoint - frame2->getWorldPos()).norm();

        if (worldDist1 == 0 || worldDist2 == 0)
        {
            badCounter++;
            continue;
        }

        float ratioOctaveScale = octaveScale2 / octaveScale1;
        float ratioWorldDist = worldDist1 / worldDist2;
        float ratioFactor = 1.5f;

        if (ratioWorldDist * ratioFactor < ratioOctaveScale || ratioWorldDist > ratioOctaveScale * ratioFactor)
        {
            badCounter++;
            continue;
        }

        // triangulate point
        std::shared_ptr<MapPoint>newMapPoint = std::make_shared<MapPoint>(newPoint, frame1, (*matches)[i].queryIdx);

        // Add other frame as observation
        newMapPoint->addObservation(frame2, (*matches)[i].trainIdx);

        // Add newMapPoint reference to both frames
        frame1->addAssociatedMapPoint((*matches)[i].queryIdx, newMapPoint);
        frame2->addAssociatedMapPoint((*matches)[i].trainIdx, newMapPoint);

        map->addMapPoint(newMapPoint);

        if (checkBaseline)
        {
            m_recentlyAddedPoints.push_back(newMapPoint);
        }
    }
    // std::cout << "Sorted out: " << badCounter << " points of " << matches->size() << endl;
}

cv::Mat SfMHelper::getCVProjectionMatrix(std::shared_ptr<Frame>frame)
{

    // OpenCv projection matrix for point triangulation needs to be 3x4
    Eigen::Matrix4f extr = frame->getPose().inverse();
    Matrix<float, 3, 4> proj = extr.block(0, 0, 3, 4);
    proj = frame->getIntrinsics() * proj;

    cv::Mat res;
    cv::eigen2cv(proj, res);

    return res;
}

Eigen::Matrix4f SfMHelper::getPoseFromConstantSpeed(SceneMap *map)
{
    // Get the key frames from the map
    std::vector<std::shared_ptr<Frame>> frames = map->getKeyFrames();
    int nbOfPoses = frames.size();
    int nbOfPosesToUse = 1;

    // Ensure there are enough frames to compute the differences
    if (nbOfPoses < 2)
    {
        return Eigen::Matrix4f::Identity();
    }

    // Calculate differences between consecutive poses
    std::vector<Eigen::Vector3f> translations;
    std::vector<Eigen::Quaternionf> rotations;
    for (int i = 0; i < nbOfPosesToUse; ++i)
    {
        Eigen::Matrix4f prevPose = frames[nbOfPoses - 1 - i - 1]->getPose();
        Eigen::Matrix4f currentPose = frames[nbOfPoses - 1 - i]->getPose();

        // Decompose matrices into translation and rotation
        Eigen::Vector3f t1 = prevPose.block<3, 1>(0, 3);
        Eigen::Matrix3f R1 = prevPose.block<3, 3>(0, 0);
        Eigen::Vector3f t2 = currentPose.block<3, 1>(0, 3);
        Eigen::Matrix3f R2 = currentPose.block<3, 3>(0, 0);

        // Compute the translation difference
        Eigen::Vector3f transDiff = R1.inverse() * (t2 - t1);
        translations.push_back(transDiff);

        // Compute the rotation difference
        Eigen::Matrix3f R_diff = R1.inverse() * R2;
        Eigen::Quaternionf diffQ(R_diff);
        rotations.push_back(diffQ);
    }

    // Initialize average difference in pose (translation + rotation)
    Eigen::Vector3f avgTranslation(0, 0, 0);
    Eigen::Quaternionf avgQuaternion(0, 0, 0, 0);

    // Add all rotations
    for (const auto &q : rotations)
    {
        // Convert rotation matrix to quaternion for better averaging of poses
        avgQuaternion.x() += q.x();
        avgQuaternion.y() += q.y();
        avgQuaternion.z() += q.z();
        avgQuaternion.w() += q.w();
    }

    // Add all translations
    for (const auto &t : translations)
    {
        avgTranslation += t;
    }

    int nbOfDifferences = rotations.size();
    // Calculate average translation and rotation
    avgTranslation /= nbOfDifferences;
    avgQuaternion.coeffs() /= nbOfDifferences;
    avgQuaternion.normalize();

    // Convert average quaternion back to a rotation matrix
    Eigen::Matrix3f avgRotation = avgQuaternion.toRotationMatrix();

    // Construct the average difference matrix
    Eigen::Matrix4f avgDiff = Eigen::Matrix4f::Identity();
    avgDiff.block<3, 3>(0, 0) = avgRotation;
    avgDiff.block<3, 1>(0, 3) = avgTranslation;

    // Get last pose
    Eigen::Matrix4f lastPose = frames.back()->getPose();

    // Construct the predicted next pose
    Eigen::Matrix4f predictedNextPose = lastPose * avgDiff;

    return predictedNextPose;
}

void SfMHelper::cullRecentMapPoints(std::shared_ptr<Frame>currentFrame, SceneMap *map)
{
    std::vector<std::shared_ptr<MapPoint>> updatedList;

    for (int i = 0; i < m_recentlyAddedPoints.size(); i++)
    {
        std::shared_ptr<MapPoint>mapPoint = m_recentlyAddedPoints[i];

        if (mapPoint->isInvalid())
        {
            continue;
        }
        else if ((currentFrame->getID() - mapPoint->getReferenceFrame()->getID()) >= 4 && mapPoint->getNumObserved() <= 2)
        {
            mapPoint->erase();
            map->eraseMapPoint(mapPoint);
            continue;
        }
        else if ((currentFrame->getID() - mapPoint->getReferenceFrame()->getID()) >= 5)
        {
            continue;
        }
        else
        {
            updatedList.push_back(mapPoint);
        }
    }

    m_recentlyAddedPoints = updatedList;
}

void SfMHelper::cullRedundantKeyframes(std::shared_ptr<Frame>currentFrame, SceneMap *map)
{
    // cull frames which have over 90% associated mappoints that are observed by at least 3 other frames (within one octave interval).
    std::vector<std::shared_ptr<Frame>> neighbors = currentFrame->getAllCovisibilityFrames();
    int counter = 0;
    for (int i = 0; i < neighbors.size(); i++)
    {
        std::shared_ptr<Frame>frame = neighbors[i];

        if (frame->getID() == 0 || frame->getID() == 1)
        {
            continue;
        }

        std::vector<std::shared_ptr<MapPoint>> mapPoints = frame->getMapPoints();

        int numberRedundantMapPoints = 0;
        int numberMapPoints = 0;

        for (int j = 0; j < mapPoints.size(); j++)
        {
            std::shared_ptr<MapPoint>mapPoint = mapPoints[j];

            if (mapPoint)
            {
                if (!mapPoint->isInvalid())
                {
                    numberMapPoints++;

                    if (mapPoint->getNumObserved() > 3)
                    {

                        int octave = frame->getKeypoint(j)->octave;
                        int numValidObservations = 0;
                        std::map<std::shared_ptr<Frame>, size_t> observingKeyframes = mapPoint->getObservingKeyframes();

                        for (auto it = observingKeyframes.begin(); it != observingKeyframes.end(); it++)
                        {

                            std::shared_ptr<Frame>observingFrame = it->first;

                            if (observingFrame == frame)
                                continue;

                            int observedOctave = observingFrame->getKeypoint(it->second)->octave;

                            // only consider frames which observe the point with the same or a smaller octave (further away)
                            if (observedOctave <= octave + 1)
                            {
                                numValidObservations++;
                                if (numValidObservations >= 3)
                                {
                                    // if at mappoint is observed by at least 3 other frames consider it as redundant observation.
                                    numberRedundantMapPoints++;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (numberMapPoints * 0.95 < numberRedundantMapPoints)
        {
            // cull frame
            frame->erase();
            map->eraseKeyFrame(frame);
            counter++;
        }
    }
    // cout << "Culled " << counter << " frames" << endl;
}

void SfMHelper::eraseOutlier(SceneMap* map, int currentFrameID)
{
    auto frames = map->getKeyFrames();
    auto points = map->getMapPoints();

    // Remove cross-reference for outliers
    for (int i = 0; i < frames.size(); i++)
    {
        int keypoints = frames[i]->getKeypointCount();
        for (int j = 0; j < keypoints; j++)
        {
            if (frames[i]->isOutlier(j))
            {

                std::shared_ptr<MapPoint> point = frames[i]->getMapPoint(j);
                if (point)
                {
                    frames[i]->eraseAssociatedMapPoint(j);
                    point->removeObservation(frames[i]);
                }
            }
        }
    }

    // remove mappoints with less than three observations and update descriptors and viewing directions
    for (int i = 0; i < points.size(); i++)
    {
        if (points[i]->isInvalid()) {
            map->eraseMapPoint(points[i]);
        }
        else {

            if (points[i]->getNumObserved() <= 2 && (currentFrameID - points[i]->getReferenceFrame()->getID()) >= 4)
            {
                points[i]->erase();
                map->eraseMapPoint(points[i]);
            }
            else
            {
                points[i]->computeDescriptor();
                points[i]->computeViewingDirection();
            }
        }
    }
}
