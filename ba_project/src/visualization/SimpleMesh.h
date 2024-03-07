#pragma once

#include <iostream>
#include <fstream>

#include "../utils/Eigen.h"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>

#include "../model/SceneMap.h"
#include "../data/VirtualSensor.h"

enum FacesType
{
	Poisson,
	GreedyProjectionTriangulation,
	None
};

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// Position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// Color stored as 4 unsigned char
	Vector4uc color;
};

struct Triangle
{
	unsigned int idx0;
	unsigned int idx1;
	unsigned int idx2;

	Triangle() : idx0{0}, idx1{0}, idx2{0} {}

	Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2) : idx0(_idx0), idx1(_idx1), idx2(_idx2) {}
};

class SimpleMesh
{
public:
	SimpleMesh(FacesType facesType = FacesType::GreedyProjectionTriangulation) : facesType(facesType){};

	// creates a mesh from a scene map
	void createMesh(SceneMap *map);

	void clear();

	// writes the mesh to a file
	bool writeMesh(const std::string &filename);

	// Getters and setters for mesh attributes
	std::vector<Vertex> &getVertices();
	const std::vector<Vertex> &getVertices() const;
	std::vector<Triangle> &getTriangles();
	const std::vector<Triangle> &getTriangles() const;
	unsigned int addVertex(Vertex &vertex);
	unsigned int addFace(unsigned int idx0, unsigned int idx1, unsigned int idx2);

private:
	std::vector<Vertex> m_vertices;
	std::vector<Triangle> m_triangles;
	std::vector<float> m_xs;
	std::vector<float> m_ys;
	std::vector<float> m_zs;
	FacesType facesType;

	float distance(Vertex &a, Vertex &b);

	// checks if a face is valid
	bool isTriangleValid(Vertex &a, Vertex &b, Vertex &c, float threshold);

	/**
	 * Generates a camera object with a given pose.
	 */
	SimpleMesh camera(const Matrix4f &normalizeMat, const Matrix4f &cameraPose, double scale = 1.f, Vector4uc color = {255, 0, 0, 255});

	/**
	 * Joins two meshes together by putting them into the common mesh and transforming the vertex positions of
	 * mesh1 with transformation 'pose1to2'.
	 */
	SimpleMesh joinMeshes(const SimpleMesh &mesh1, const SimpleMesh &mesh2, Matrix4f pose1to2 = Matrix4f::Identity());

	// Uses greedy projection triangulation to create faces for the pointcloud
	void createMeshFromGreedyProjectionTriangulation();

	// Uses poisson to create faces for the pointcloud
	void createMeshFromPoisson();

	// Gets the color for a map point based on the corresponding color in the frame
	Vector4uc getColorForMapPoint(std::shared_ptr<MapPoint>mapPoint);
};
