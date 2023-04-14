#pragma once

#include <Eigen/Dense>
#include "../utils/mypcl_cloud_type.h"
namespace ctsm{
    
struct DynePoint{
    Eigen::Vector3d point;
    float velocity;
    double timestamp;
};//

RTVPointCloud vector2pointcloud(std::vector<DynePoint> & vector){
    RTVPointCloud tempPointCloud;
    tempPointCloud.resize(vector.size());
    for(auto point : vector)
    {
        RTVPoint point_;
        point_.x = point.point.x();
        point_.y = point.point.y();
        point_.z = point.point.z();
        point_.time = point.timestamp;
        point_.velocity = point.velocity;
        tempPointCloud.push_back(point_);
    }
    return tempPointCloud;
}


}