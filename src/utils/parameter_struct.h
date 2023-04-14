#pragma once

#include <glog/logging.h>
#include "yaml_utils.h"
#include "sophus_utils.hpp"

namespace ctsm{

#define RESET "\033[0m"
#define BLACK "\033[30m"   /* Black */
#define RED "\033[31m"     /* Red */
#define GREEN "\033[32m"   /* Green */
#define YELLOW "\033[33m"  /* Yellow */
#define BLUE "\033[34m"    /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m"    /* Cyan */
#define WHITE "\033[37m"   /* White */

extern double GRAVITY_NORM;

enum MODE {
  Odometry_Offline = 1,  //
  Odometry_Online,       //
};




struct StateVector{
    StateVector() : timestamp(0){}

    double timestamp;
    Eigen::Vector3d p; //global frame
    Eigen::Vector3d v; //global frame
    Eigen::Vector3d q;
};




}