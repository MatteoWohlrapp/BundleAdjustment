#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "../model/Frame.h"
#include "../model/SceneMap.h"
#include "SfMHelper.h"

struct ReprojectionError
{
    ReprojectionError(Vector2f observedPos, Matrix3f cameraIntrinsics)
        : observed_pos(observedPos), cameraIntrinsics(cameraIntrinsics) {}

    template <typename T>
    bool operator()(const T *const camera, const T *const point, T *residuals) const
    {
        const Eigen::Matrix<T, 3, 4> extr(camera);
        const Eigen::Matrix<T, 3, 3> rot = extr.block(0, 0, 3, 3);
        const Eigen::Matrix<T, 4, 1> pos{point[0], point[1], point[2], T(1.0)};

        const Eigen::Matrix<T, 3, 1> p = extr * pos;

        const Eigen::Matrix<T, 2, 1> predicted_pixel = (cameraIntrinsics.cast<T>() * p).hnormalized();

        // The error is the difference between the predicted and observed position.
        const Eigen::Matrix<T, 2, 1> res = predicted_pixel - observed_pos.cast<T>();

        residuals[0] = res[0];
        residuals[1] = res[1];
        // apply loss based on characteristics of the rotation matrix: 1. det(rot)=1 2. rot.transpose() =rot.inverse()
        residuals[2] = ceres::abs(1.0 - rot.determinant());
        residuals[3] = (rot.transpose() - rot.inverse()).template lpNorm<1>();
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const Vector2f observedPos, const Matrix3f cameraIntrinsics)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 4, 12, 3>(
            new ReprojectionError(observedPos, cameraIntrinsics)));
    }

    Vector2f observed_pos;
    Matrix3f cameraIntrinsics;
};

// adapted from http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
struct AngleReprojectionError
{
    AngleReprojectionError(Vector2f observedPos, Matrix3f cameraIntrinsics)
        : observed_pos(observedPos), cameraIntrinsics(cameraIntrinsics) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {

        T rotMat[9];
        ceres::AngleAxisToRotationMatrix(camera, rotMat);
        Matrix<T, 3, 3> MatE(rotMat);

        Eigen::Matrix<T, 3, 4> extr;
        extr.block(0, 0, 3, 3) = MatE;
        extr.block(0, 3, 3, 1) = Matrix<T, 3, 1>(&camera[3]);

        const Eigen::Matrix<T, 4, 1> pos{point[0], point[1], point[2], T(1.0)};
        const Eigen::Matrix<T, 3, 1> p = extr * pos;
        const Eigen::Matrix<T, 2, 1> predicted_pixel = (cameraIntrinsics.cast<T>() * p).hnormalized();
        // The error is the difference between the predicted and observed position.
        const Eigen::Matrix<T, 2, 1> res = predicted_pixel - observed_pos.cast<T>();
        residuals[0] = res[0];
        residuals[1] = res[1];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const Vector2f observedPos, const Matrix3f cameraIntrinsics)
    {
        return (new ceres::AutoDiffCostFunction<AngleReprojectionError, 2, 6, 3>(
            new AngleReprojectionError(observedPos, cameraIntrinsics)));
    }

    Vector2f observed_pos;
    Matrix3f cameraIntrinsics;
};


struct PointOnlyReprojectionError
{
    PointOnlyReprojectionError(Vector2f observedPos, Matrix4f cameraExtrinsics, Matrix3f cameraIntrinsics)
        : observed_pos(observedPos), extr(cameraExtrinsics), cameraIntrinsics(cameraIntrinsics) {}

    template <typename T>
    bool operator()(const T* const point, T* residuals) const
    {
        const Eigen::Matrix<T, 4, 1> pos{ point[0], point[1], point[2], T(1.0) };
        const Eigen::Matrix<T, 3, 1> p = (extr.cast<T>() * pos).hnormalized();
        const Eigen::Matrix<T, 2, 1> predicted_pixel = (cameraIntrinsics.cast<T>() * p).hnormalized();
        // The error is the difference between the predicted and observed position.
        const Eigen::Matrix<T, 2, 1> res = predicted_pixel - observed_pos.cast<T>();
        residuals[0] = res[0];
        residuals[1] = res[1];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Vector2f observedPos, const Matrix4f cameraExtrinsics, const Matrix3f cameraIntrinsics)
    {
        return (new ceres::AutoDiffCostFunction<PointOnlyReprojectionError, 2, 3>(
            new PointOnlyReprojectionError(observedPos,cameraExtrinsics, cameraIntrinsics)));
    }

    Vector2f observed_pos;
    Matrix4f extr;
    Matrix3f cameraIntrinsics;
};

struct PoseOnlyReprojectionError
{
    PoseOnlyReprojectionError(Vector2f observedPos, Matrix3f cameraIntrinsics, Vector4f worldPosh)
        : observed_pos(observedPos), cameraIntrinsics(cameraIntrinsics), world_posh(worldPosh)
    {
    }

    template <typename T>
    bool operator()(const T *const camera, T *residuals) const
    {

        const Matrix<T, 3, 4> extr(camera);
        const Eigen::Matrix<T, 3, 1> p = (extr * world_posh.cast<T>());

        const Eigen::Matrix<T, 2, 1> predicted_pixel = (cameraIntrinsics.cast<T>() * p).hnormalized();
        const Eigen::Matrix<T, 2, 1> res = predicted_pixel - observed_pos.cast<T>();

        residuals[0] = res[0];
        residuals[1] = res[1];
        return true;
    }

    static ceres::CostFunction *Create(const Vector2f observedPos, const Matrix3f cameraIntrinsics, const Vector4f woldPosh)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 4, 12>(
            new PoseOnlyReprojectionError(observedPos, cameraIntrinsics, woldPosh)));
    }

private:
    Vector2f observed_pos;
    Matrix3f cameraIntrinsics;
    Vector4f world_posh;
};

struct PoseOnlyAngleReprojectionError
{
    PoseOnlyAngleReprojectionError(Vector2f observedPos, Matrix3f cameraIntrinsics, Vector4f worldPosh)
        : observed_pos(observedPos), cameraIntrinsics(cameraIntrinsics), world_posh(worldPosh)
    {
    }

    template <typename T>
    bool operator()(const T *const camera, T *residuals) const
    {
        T rotMat[9];
        ceres::AngleAxisToRotationMatrix(camera, rotMat);
        Matrix<T, 3, 3> MatE(rotMat);

        Eigen::Matrix<T, 3, 4> extr;
        extr.block(0, 0, 3, 3) = MatE;
        extr.block(0, 3, 3, 1) = Matrix<T, 3, 1>(&camera[3]);

        const Eigen::Matrix<T, 3, 1> p = (extr * world_posh.cast<T>());

        const Eigen::Matrix<T, 2, 1> predicted_pixel = (cameraIntrinsics.cast<T>() * p).hnormalized();
        const Eigen::Matrix<T, 2, 1> res = predicted_pixel - observed_pos.cast<T>();

        residuals[0] = res[0];
        residuals[1] = res[1];
        return true;
    }

    static ceres::CostFunction *Create(const Vector2f observedPos, const Matrix3f cameraIntrinsics, const Vector4f woldPosh)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyAngleReprojectionError, 2, 6>(
            new PoseOnlyAngleReprojectionError(observedPos, cameraIntrinsics, woldPosh)));
    }

private:
    Vector2f observed_pos;
    Matrix3f cameraIntrinsics;
    Vector4f world_posh;
};

/**
 * BA optimizer - Abstract Base Class
 */
class BAOptimizer
{
public:
    BAOptimizer() : m_nIterations(3),
                    m_nMaxItPerBA(200)
    {
    }

    void setNbOfIterations(unsigned nIterations)
    {
        m_nIterations = nIterations;
    }

    void setNbOfMaxItPerBA(unsigned nMaxItPerBA)
    {
        m_nMaxItPerBA = nMaxItPerBA;
    }

protected:
    unsigned m_nIterations;
    unsigned m_nMaxItPerBA;

    void pruneCorrespondences(std::shared_ptr<Frame>frame, bool considerOutlier = true);
    void configureSolver(ceres::Solver::Options &options);
};

/**
 * GlobalBA optimizer - using Ceres for optimization.
 */
class GlobalBAOptimizer : public BAOptimizer
{
public:
    GlobalBAOptimizer() {}

    void optimizeCamerasAndMapPoints(SceneMap *optimizedMap, bool eraseOutliers, int currentFrameID);

private:
    void prepareConstraints(const std::vector<std::shared_ptr<Frame>> &frames, const std::vector<std::shared_ptr<MapPoint>> &points,
                            ceres::Problem &problem, std::vector<Matrix<double, 3, 4>> &optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d> &optimizedPoints) const;
};

/**
 * GlobalBA optimizer with optimizing angles for the frame/camera-pose - using Ceres for optimization.
 */
class GlobalBAOptimizerAngles : public BAOptimizer
{
public:
    GlobalBAOptimizerAngles() {}

    void optimizeCamerasAndMapPoints(SceneMap *optimizedMap, bool eraseOutliers, int currentFrameID);

private:
    void prepareConstraints(const std::vector<std::shared_ptr<Frame>> &frames, const std::vector<std::shared_ptr<MapPoint>> &points,
                            ceres::Problem &problem, std::vector<Matrix<double, 6, 1>> &optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d> &optimizedPoints) const;
};

class MotionOnlyBAOptimizer : public BAOptimizer
{
public:
    MotionOnlyBAOptimizer() {}

    void optimizeCameraPose(std::shared_ptr<Frame>frame);

private:
    void prepareConstraints(std::shared_ptr<Frame>frame, const std::vector<std::shared_ptr<MapPoint>> &points, bool useHuberLoss,
                            ceres::Problem &problem, Matrix<double, 3, 4> &optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d> &optimizedPoints) const;
};

class MotionOnlyBAOptimizerAngles : public BAOptimizer
{
public:
    MotionOnlyBAOptimizerAngles() {}

    void optimizeCameraPose(std::shared_ptr<Frame>frame);

private:
    void prepareConstraints(std::shared_ptr<Frame>frame, const std::vector<std::shared_ptr<MapPoint>> &points,
                            bool useHuberLoss, ceres::Problem &problem, Matrix<double, 6, 1> &optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d> &optimizedPoints) const;
};

class LocalBAOptimizerAngles : public BAOptimizer
{
public:
    LocalBAOptimizerAngles() {}

    void optimizeCamerasAndMapPoints(SceneMap* optimizedMap, std::shared_ptr<Frame> currentFrame, bool eraseOutliers);

private:
    void prepareConstraints(const std::vector<std::shared_ptr<Frame>>& localFrames, const std::vector<std::shared_ptr<Frame>>& fixedFrames, const std::vector<std::shared_ptr<MapPoint>>& points,
        ceres::Problem& problem, std::vector<Matrix<double, 6, 1>>& optimizedExtr, std::map<std::shared_ptr<MapPoint>, Vector3d>& optimizedPoints) const;
};
