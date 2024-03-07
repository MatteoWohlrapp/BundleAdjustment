#include "SimpleMesh.h"

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


void SimpleMesh::createMesh(SceneMap *map)
{
	vector<std::shared_ptr<MapPoint>> map_points = map->getMapPoints();
	int map_size = map_points.size();
	// m_vertices.resize(map_size);
	int countZero = 0;
	int countOutlier = 0;

	std::vector<Vertex> inlier;
	std::vector<float> distances;
	// add vertices
	for (int idx = 0; idx < map_size; ++idx)
	{
		Vector3f position = map_points[idx]->getPosition();
		// go through map of all observing frames for the map point and check if they are outliers
		bool isOutlier = false;
		for (auto &pair : (map_points[idx]->getObservingKeyframes()))
		{
			std::shared_ptr<Frame>frame = pair.first;
			size_t indexInFrame = pair.second;
			if (frame->isOutlier(indexInFrame))
			{
				isOutlier = true;
				countOutlier++;
				break;
			}
		}

		// check if outlier and positions for x, y, z smaller than 1
		if (!isOutlier)
		{
			// get color from reference frame
			Vertex v;
			v.position = Vector4f(position.x(), position.y(), position.z(), 1.0);
			v.color = map_points[idx]->getReferenceColor();
			inlier.push_back(v);
			float dist = position.norm();
			distances.push_back(dist);
		}
	}
	//cout << "#Outlier: " << countOutlier << endl;
	//cout << "#Inlier: " << map_size - countOutlier << endl;

	/*
	// filter out largest values
	std::vector<float> dist_sorted(distances);
	std::sort(dist_sorted.begin(), dist_sorted.end());
	float max_dist = dist_sorted[std::floor(dist_sorted.size() * 0.95)];
	*/

	// for computing bounding volume
	float x_max = FLT_MIN;
	float x_min = FLT_MAX;
	float y_max = FLT_MIN;
	float y_min = FLT_MAX;
	float z_max = FLT_MIN;
	float z_min = FLT_MAX;

	for (int i = 0; i < inlier.size(); i++)
	{
		Vertex v = inlier[i];
		//if (distances[i] <= max_dist)
		//{
			m_vertices.push_back(v);

			if (v.position.x() < x_min)
			{
				x_min = v.position.x();
			}
			if (v.position.y() < y_min)
			{
				y_min = v.position.y();
			}
			if (v.position.z() < z_min)
			{
				z_min = v.position.z();
			}
			if (v.position.x() > x_max)
			{
				x_max = v.position.x();
			}
			if (v.position.y() > y_max)
			{
				y_max = v.position.y();
			}
			if (v.position.z() > z_max)
			{
				z_max = v.position.z();
			}
		//}
	}

	//std::cout << "After filtering: " << m_vertices.size() << endl;

	Vector3f maxVec, minVec;

	maxVec << x_max, y_max, z_max;
	minVec << x_min, y_min, z_min;

	Vector3f size = maxVec - minVec;
	Vector3f center = (maxVec + minVec) * 0.5;

	// scale so that smallest dimension is between 0.0 an 1.0
	float isoScale = fminf(fminf(size.x(), size.y()), size.z());
	isoScale = isoScale > 0.0 ? 1.0 / isoScale : 1.0;
	Matrix3f scale = Matrix3f::Identity();
	scale = scale * isoScale;
	//cout << scale << endl;
	Matrix4f normalizeMat = Matrix4f::Identity();
	normalizeMat.block(0, 0, 3, 3) = scale;
	normalizeMat.block(0, 3, 3, 1) = -center;

	// normalize vertices
	for (int i = 0; i < m_vertices.size(); i++)
	{
		m_vertices[i].position = (normalizeMat * m_vertices[i].position).hnormalized().homogeneous();
	}

	/*
	// write ply file with m_vertices 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	// iterate thorugh all vertices and add to pcl cloud 
	for (int i = 0; i < m_vertices.size(); i++)
	{
		Vertex v = m_vertices[i];
		pcl::PointXYZRGB point;
		point.x = v.position.x();
		point.y = v.position.y();
		point.z = v.position.z();
		point.r = v.color.x();
		point.g = v.color.y();
		point.b = v.color.z();
		coloredCloud->push_back(point);
	}

	pcl::io::savePLYFile("output_mesh.ply", *coloredCloud);*/

	if (facesType == FacesType::GreedyProjectionTriangulation)
	{
		createMeshFromGreedyProjectionTriangulation();
	}
	else if (facesType == FacesType::Poisson)
	{
		createMeshFromPoisson();
	}

	// add camera poses
	for (auto &frame : map->getKeyFrames())
	{
		// add camera pose and join with current mesh
		Matrix4f pose = frame->getPose();
		SimpleMesh cameraMesh = camera(normalizeMat, pose, (1.0 / isoScale) * 0.001, Vector4uc(255, 0, 0, 255));
		*this = joinMeshes(cameraMesh, *this);
	}
}

void SimpleMesh::clear()
{
	m_vertices.clear();
	m_triangles.clear();
}

unsigned int SimpleMesh::addVertex(Vertex &vertex)
{
	unsigned int vId = (unsigned int)m_vertices.size();
	m_vertices.push_back(vertex);
	return vId;
}

unsigned int SimpleMesh::addFace(unsigned int idx0, unsigned int idx1, unsigned int idx2)
{
	unsigned int fId = (unsigned int)m_triangles.size();
	Triangle triangle(idx0, idx1, idx2);
	m_triangles.push_back(triangle);
	return fId;
}

std::vector<Vertex> &SimpleMesh::getVertices()
{
	return m_vertices;
}

const std::vector<Vertex> &SimpleMesh::getVertices() const
{
	return m_vertices;
}

std::vector<Triangle> &SimpleMesh::getTriangles()
{
	return m_triangles;
}

const std::vector<Triangle> &SimpleMesh::getTriangles() const
{
	return m_triangles;
}

bool SimpleMesh::writeMesh(const std::string &filename)
{
	// Write off file.
	std::ofstream outFile(filename);
	if (!outFile.is_open())
		return false;

	// Write header.
	outFile << "COFF" << std::endl;
	outFile << m_vertices.size() << " " << m_triangles.size() << " 0" << std::endl;

	// Save vertices.
	for (unsigned int i = 0; i < m_vertices.size(); i++)
	{
		const auto &vertex = m_vertices[i];
		if (vertex.position.allFinite())
		{
			outFile << vertex.position.x() << " " << vertex.position.y() << " " << vertex.position.z() << " " << int(vertex.color.x()) << " " << int(vertex.color.y()) << " " << int(vertex.color.z()) << " " << int(vertex.color.w()) << std::endl;
		}
		else
		{
			outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
		}
	}

	// Save faces.
	for (unsigned int i = 0; i < m_triangles.size(); i++)
	{
		outFile << "3 " << m_triangles[i].idx0 << " " << m_triangles[i].idx1 << " " << m_triangles[i].idx2 << std::endl;
	}

	// Close file.
	outFile.close();

	return true;
}

float SimpleMesh::distance(Vertex &a, Vertex &b)
{
	return sqrt(pow(a.position.x() - b.position.x(), 2) +
				pow(a.position.y() - b.position.y(), 2) +
				pow(a.position.z() - b.position.z(), 2));
}

bool SimpleMesh::isTriangleValid(Vertex &a, Vertex &b, Vertex &c, float threshold)
{
	// Check if all vertices are valid
	if (a.position.x() == MINF || a.position.y() == MINF || a.position.z() == MINF ||
		b.position.x() == MINF || b.position.y() == MINF || b.position.z() == MINF ||
		c.position.x() == MINF || c.position.y() == MINF || c.position.z() == MINF)
	{
		return false;
	}

	// Check distances with Euclidean norm
	if (distance(a, b) > threshold ||
		distance(b, c) > threshold ||
		distance(c, a) > threshold)
	{
		return false;
	}

	return true;
}

SimpleMesh SimpleMesh::camera(const Matrix4f &normalizeMat, const Matrix4f &cameraPose, double scale, Vector4uc color)
{
	SimpleMesh mesh = SimpleMesh();
	Matrix4f cameraToWorld = cameraPose;

	// These are precomputed values for sphere aproximation.
	std::vector<double> vertexComponents = {25, 25, 0, -50, 50, 100, 49.99986, 49.9922, 99.99993, -24.99998, 25.00426, 0.005185,
											25.00261, -25.00023, 0.004757, 49.99226, -49.99986, 99.99997, -50, -50, 100, -25.00449, -25.00492, 0.019877};
	const std::vector<unsigned> faceIndices = {1, 2, 3, 2, 0, 3, 2, 5, 4, 4, 0, 2, 5, 6, 7, 7, 4, 5, 6, 1, 7, 1, 3, 7, 3, 0, 4, 7, 3, 4, 5, 2, 1, 5, 1, 6};

	// Add vertices.
	for (int i = 0; i < 8; ++i)
	{
		Vertex v;
		Vector4f camCords;
		camCords << (float)(scale * vertexComponents[3 * i + 0]), (float)(scale * vertexComponents[3 * i + 1]), (float)(scale * vertexComponents[3 * i + 2]), 1.f;
		v.position = (normalizeMat * cameraToWorld * camCords).hnormalized().homogeneous();
		v.color = color;
		mesh.addVertex(v);
	}

	// Add faces.
	for (int i = 0; i < 12; ++i)
	{
		mesh.addFace(faceIndices[3 * i + 0], faceIndices[3 * i + 1], faceIndices[3 * i + 2]);
	}

	return mesh;
}

SimpleMesh SimpleMesh::joinMeshes(const SimpleMesh &mesh1, const SimpleMesh &mesh2, Matrix4f pose1to2)
{
	SimpleMesh joinedMesh;
	const auto &vertices1 = mesh1.getVertices();
	const auto &triangles1 = mesh1.getTriangles();
	const auto &vertices2 = mesh2.getVertices();
	const auto &triangles2 = mesh2.getTriangles();

	auto &joinedVertices = joinedMesh.getVertices();
	auto &joinedTriangles = joinedMesh.getTriangles();

	const unsigned nVertices1 = vertices1.size();
	const unsigned nVertices2 = vertices2.size();
	joinedVertices.reserve(nVertices1 + nVertices2);

	const unsigned nTriangles1 = triangles1.size();
	const unsigned nTriangles2 = triangles2.size();
	joinedTriangles.reserve(nVertices1 + nVertices2);

	// Add all vertices (we need to transform vertices of mesh 1).
	for (int i = 0; i < nVertices1; ++i)
	{
		const auto &v1 = vertices1[i];
		Vertex v;
		v.position = pose1to2 * v1.position;
		v.color = v1.color;
		joinedVertices.push_back(v);
	}
	for (int i = 0; i < nVertices2; ++i)
		joinedVertices.push_back(vertices2[i]);

	// Add all faces (the indices of the second mesh need to be added an offset).
	for (int i = 0; i < nTriangles1; ++i)
		joinedTriangles.push_back(triangles1[i]);
	for (int i = 0; i < nTriangles2; ++i)
	{
		const auto &t2 = triangles2[i];
		Triangle t{t2.idx0 + nVertices1, t2.idx1 + nVertices1, t2.idx2 + nVertices1};
		joinedTriangles.push_back(t);
	}

	return joinedMesh;
}

void SimpleMesh::createMeshFromGreedyProjectionTriangulation()
{
	// add triangles using poisson surface reconstrunction
	// approach adapted from here: https://pcl.readthedocs.io/projects/tutorials/en/latest/greedy_projection.html
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Convert your map points to pcl::PointXYZ format
	for (const auto &vertex : m_vertices)
	{
		cloud->push_back(pcl::PointXYZ(vertex.position.x(), vertex.position.y(), vertex.position.z()));
	}

	// Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals);

	// Initialize objects
	//pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	auto *gp3 = new pcl::GreedyProjectionTriangulation<pcl::PointNormal>;
	pcl::PolygonMesh triangles;

	// Set the maximum distance between connected points (maximum edge length)
	gp3->setSearchRadius(25);

	// Set typical values for the parameters
	gp3->setMu(2.5);
	gp3->setMaximumNearestNeighbors(100);
	gp3->setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
	gp3->setMinimumAngle(M_PI / 18);	   // 10 degrees
	gp3->setMaximumAngle(2 * M_PI / 3);	   // 120 degrees
	gp3->setNormalConsistency(false);

	// Get result
	gp3->setInputCloud(cloud_with_normals);
	gp3->setSearchMethod(tree2);
	gp3->reconstruct(triangles);

	// print size of polygon
	std::cout << "Number of polygons: " << triangles.polygons.size() << std::endl;
	// Iterate over each polygon in the PCL mesh
	for (const auto &polygon : triangles.polygons)
	{

		if (polygon.vertices.size() >= 3)
		{
			// Add each face to the SimpleMesh
			// check if vertices are valid
			if (isTriangleValid(m_vertices[polygon.vertices[0]], m_vertices[polygon.vertices[1]], m_vertices[polygon.vertices[2]], 4.0f))
			{
				this->addFace(polygon.vertices[0], polygon.vertices[1], polygon.vertices[2]);
			}
		}
	}
}

void SimpleMesh::createMeshFromPoisson()
{
	// add triangles using poisson surface reconstrunction
	// approach adapted from here: https://pcl.readthedocs.io/projects/tutorials/en/latest/greedy_projection.html
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	// Convert your map points to pcl::PointXYZ format
	for (int i = 0; i < m_vertices.size(); i++)
	{
		Vertex vertex = m_vertices[i];
		cloud->push_back(pcl::PointXYZRGB(vertex.position.x(), vertex.position.y(), vertex.position.z(), vertex.color.x(), vertex.color.y(), vertex.color.z()));
	}

	// Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

	pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
	poisson.setDepth(9); // You can adjust this parameter
	pcl::PolygonMesh triangles;

	// Set the input cloud
	poisson.setInputCloud(cloud_with_normals);

	// Perform reconstruction
	// poisson.reconstruct(triangles);
	poisson.performReconstruction(triangles);

	// print size of polygon
	std::cout << "Number of polygons: " << triangles.polygons.size() << std::endl;
	// Iterate over each polygon in the PCL mesh

	pcl::PointCloud<pcl::PointXYZ>::Ptr allVertices(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromPCLPointCloud2(triangles.cloud, *allVertices);

	// clear before written points
	m_vertices.clear();
	m_vertices.reserve(allVertices->size());
	// add vertices
	for (int i = 0; i < allVertices->size(); i++)
	{
		Vertex v;
		v.position = Vector4f(allVertices->points[i].x, allVertices->points[i].y, allVertices->points[i].z, 1.0);
		// x.color = Vector4uc(allVertices->points[i].r, allVertices->points[i].g, allVertices->points[i].b, 255);
		v.color = Vector4uc(255, 255, 0, 255);
		m_vertices.push_back(v);
	}

	int countFaces = 0;
	for (int i = 0; i < triangles.polygons.size(); i++)
	{
		pcl::Vertices &polygon = triangles.polygons[i];
		if (polygon.vertices.size() >= 3)
		{
			// Add each face to the SimpleMesh
			// check if vertices are valid
			if (polygon.vertices[0] < m_vertices.size() && polygon.vertices[1] < m_vertices.size() && polygon.vertices[2] < m_vertices.size())
			{
				if (isTriangleValid(m_vertices[polygon.vertices[0]], m_vertices[polygon.vertices[1]], m_vertices[polygon.vertices[2]], 4.0f))
				{
					this->addFace(polygon.vertices[0], polygon.vertices[1], polygon.vertices[2]);
					countFaces++;
				}
			}
		}
	}
	cout << "#Faces added: " << countFaces << endl;
}

Vector4uc SimpleMesh::getColorForMapPoint(std::shared_ptr<MapPoint>mapPoint)
{
	std::shared_ptr<Frame>frame = mapPoint->getReferenceFrame();
	// Get the 2D keypoint position for this map point in the frame
	Vector2f keypointPosition = mapPoint->getCorresponding2DKeyPointPosition(frame);

	// Access the color image from the frame
	cv::Mat colorImage = frame->getColor();

	// Ensure the keypoint position is within the image bounds
	if (keypointPosition.x() >= 0 && keypointPosition.x() < colorImage.cols &&
		keypointPosition.y() >= 0 && keypointPosition.y() < colorImage.rows)
	{
		// Retrieve the color at the keypoint position

		// pixel coordinates are subpixel, so we have to extract subpixel value
		cv::Mat interpolated;
		cv::getRectSubPix(colorImage, cv::Size(1, 1), cv::Point2f(keypointPosition.x(), keypointPosition.y()), interpolated);
		cv::Vec3b color = interpolated.at<cv::Vec3b>(0, 0);

		// Convert the color to Vector4uc format and return
		return Vector4uc(color[0], color[1], color[2], 255); // Assuming full opacity
	}
	else
	{
		// Return a default color (e.g., black) if the position is out of bounds
		return Vector4uc(0, 0, 0, 128);
	}
}
