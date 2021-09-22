// Author of MMS_SLAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#ifndef _LASER_MAPPING_H_
#define _LASER_MAPPING_H_

//PCL lib
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/filters/radius_outlier_removal.h>

//eigen  lib
#include <Eigen/Dense>
#include <Eigen/Geometry>

//c++ lib
#include <string>
#include <math.h>
#include <vector>


#define LASER_CELL_WIDTH 10.0
#define LASER_CELL_HEIGHT 10.0
#define LASER_CELL_DEPTH 10.0

//separate map as many sub point clouds

#define LASER_CELL_RANGE_HORIZONTAL 2
#define LASER_CELL_RANGE_VERTICAL 2

#define MIN_MAP_UPDATE_FRAME 6
#define MIN_MAP_UPDATE_ANGLE 30
#define MIN_MAP_UPDATE_DISTANCE 1.0

class LaserMappingClass 
{

    public:
    	LaserMappingClass();
		void init(double map_resolution);
		void updateCurrentPointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_in, const Eigen::Isometry3d& pose_current);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr getMap(void);

	private:
		int origin_in_map_x;
		int origin_in_map_y;
		int origin_in_map_z;
		int map_width;
		int map_height;
		int map_depth;
		std::vector<std::vector<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>>> map;
		pcl::VoxelGrid<pcl::PointXYZRGB> downSizeFilter;
		pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> noise_filter;
		void addWidthCellNegative(void);
		void addWidthCellPositive(void);
		void addHeightCellNegative(void);
		void addHeightCellPositive(void);
		void addDepthCellNegative(void);
		void addDepthCellPositive(void);
		void checkPoints(int& x, int& y, int& z);

};


#endif // _LASER_MAPPING_H_

