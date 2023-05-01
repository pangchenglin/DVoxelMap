#pragma once

#include <glog/logging.h>

#include "../utils/mypcl_cloud_type.h"
#include "../utils/parameter_struct.h"
#include "se3_spline.h"

using namespace clic;

class TrajectoryManager;

class Trajectory : public Se3Spline<SplineOrder, double>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Trajectory> Ptr;
    
    static constexpr double NS_TO_S = 1e-9;  ///< Nanosecond to second conversion
    static constexpr double S_TO_NS = 1e9;   ///< Second to nanosecond conversion

    Trajectory(double time_interval,double start_time = 0)
    : Se3Spline<SplineOrder,double>(time_interval * S_TO_NS, start_time * S_TO_NS),
    data_start_time_(-1),active_time_(-1),forced_fixed_time_(-1)
    {
        this->extendKnotsTo(start_time * S_TO_NS, SO3d(Eigen::Quaterniond::Identity()),Eigen::Vector3d(0,0,0));
    }


    double GetDataStartTime() const {return data_start_time_;}

    double GetActiveTime() const {return active_time_;};

    double GetForchdFixedTime() const {return forced_fixed_time_;}

    double minTime() const {
        return NS_TO_S * this->minTimeNs();
    }

    double maxTime() const {
        return NS_TO_S * this->maxTimeNs();
    }

    Eigen::Vector3d GetPositionWorld(const double timestamp) const {
        return this->positionWorld(timestamp * S_TO_NS);
    }

    Eigen::Vector3d GetTransVelWorld(const double timestamp) const {
        return this->transVelWorld(timestamp * S_TO_NS);
    }

    Eigen::Vector3d GetTransAccelWorld(const double timestamp) const {
        return this->transAccelWorld(timestamp * S_TO_NS);
    }

    Eigen::Vector3d GetRotVelBody(const double timestamp) const {
        return this->rotVelBody(timestamp * S_TO_NS);
    }

    void SetForcedFixedTime(double time){
        if(time < minTime())
            forced_fixed_time_ = minTime();
        else
            forced_fixed_time_ = time;
    }

    void SetActiveTime(double time){active_time_ = time;}

    

private:
  double data_start_time_;  
  double active_time_;       
  double forced_fixed_time_;  

  std::map<SensorType, ExtrinsicParam> EP_StoI_;

  friend TrajectoryManager;
};//class Trajectory; 