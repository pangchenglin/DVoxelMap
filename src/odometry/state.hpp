#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include "../utils/mypcl_cloud_type.h"
#include "../utils/so3_math.h"
#include "../utils/eigen_utils.hpp"
// #define INIT_COV (0.0000001)
// #define DIM_STATE (15)  // Dimension of states (Let Dim(SO(3)) = 3)

struct StatesGroup{
    StatesGroup(){
        this->rot = Eigen::Matrix3d::Identity();
        this->pos = Eigen::Vector3d::Zero();
        this->vel = Eigen::Vector3d::Zero();
        this->bias_a = Eigen::Vector3d::Zero();
        this->bias_g = Eigen::Vector3d::Zero();
        this->cov = Eigen::Matrix<double,DIM_STATE,DIM_STATE>::Identity() * INIT_COV;
    }


  StatesGroup(const StatesGroup &b) {
    this->rot = b.rot;
    this->pos = b.pos;
    this->vel = b.vel;
    this->bias_a = b.bias_a;
    this->bias_g = b.bias_g;
    this->cov = b.cov;
  };

  StatesGroup &operator=(const StatesGroup &b) {
    this->rot = b.rot;
    this->pos = b.pos;
    this->vel = b.vel;
    this->cov = b.cov;
    this->bias_a = b.bias_a;
    this->bias_g = b.bias_g;
    return *this;
  };

    StatesGroup operator+(const Eigen::Matrix<double, DIM_STATE, 1> &state_add) {
    StatesGroup a;
    a.rot = this->rot * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    a.pos = this->pos + state_add.block<3, 1>(3, 0);
    a.vel = this->vel + state_add.block<3, 1>(6, 0);
    a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
    a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);
    a.cov = this->cov;
    return a;
    };

    StatesGroup &operator+=(const Matrix<double, DIM_STATE, 1> &state_add) {
    this->rot =
    this->rot * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    this->pos += state_add.block<3, 1>(3, 0);
    this->vel += state_add.block<3, 1>(6, 0);
    this->bias_g += state_add.block<3, 1>(9, 0);
    this->bias_a += state_add.block<3, 1>(12, 0);
    return *this;
    };

    Matrix<double, DIM_STATE, 1> operator-(const StatesGroup &b) {
    Matrix<double, DIM_STATE, 1> a;
    Eigen::Matrix3d rotd(b.rot.transpose() * this->rot);
    a.block<3, 1>(0, 0) = Log(rotd);
    a.block<3, 1>(3, 0) = this->pos - b.pos;
    a.block<3, 1>(6, 0) = this->vel - b.vel;
    a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
    a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
    return a;
    };

    void resetpose() {
    this->rot = Eigen::Matrix3d::Identity();
    this->pos = Eigen::Vector3d::Zero();
    this->vel = Eigen::Vector3d::Zero();
    }

    Eigen::Matrix3d rot;
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Eigen::Vector3d bias_a;
    Eigen::Vector3d bias_g;

    Eigen::Matrix<double,DIM_STATE,DIM_STATE> cov;
};
/************************esti_normvector*******************************/
template <typename T>
bool esti_normvector(Eigen::Matrix<T, 3, 1> &normvec, const PointVector &point,
                     const T &threshold, const int &point_num) {
  MatrixXf A(point_num, 3);
  MatrixXf b(point_num, 1);
  b.setOnes();
  b *= -1.0f;

  for (int j = 0; j < point_num; j++) {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }
  normvec = A.colPivHouseholderQr().solve(b);

  for (int j = 0; j < point_num; j++) {
    if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y +
             normvec(2) * point[j].z + 1.0f) > threshold) {
      return false;
    }
  }

  normvec.normalize();
  return true;
}

template <typename T>
bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point,
                const T &threshold) {
  Matrix<T, NUM_MATCH_POINTS, 3> A;
  Matrix<T, NUM_MATCH_POINTS, 1> b;
  b.setOnes();
  b *= -1.0f;

  for (int j = 0; j < NUM_MATCH_POINTS; j++) {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }

  Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

  for (int j = 0; j < NUM_MATCH_POINTS; j++) {
    if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y +
             normvec(2) * point[j].z + 1.0f) > threshold) {
      return false;
    }
  }

  T n = normvec.norm();
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;
  return true;
}

/************************Process************************/
class State_Process{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    State_Process();
    ~State_Process();

    void Reset();

    void only_propag(const RTVPointCloud::Ptr &meas, const double &timestamp, StatesGroup &state_input);
    void set_acc_cov_scale(const V3D &scalar);
    void set_gyr_cov_scale(const V3D &scalar);

    V3D cov_acc;
    V3D cov_gyr;
    V3D cov_acc_scale;
    V3D cov_gyr_scale;
private:

    bool b_first_frame = true;
    double time_last_scan;
};

State_Process::State_Process()
: b_first_frame(true){
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  cov_acc_scale = V3D(1, 1, 1);
  cov_gyr_scale = V3D(1, 1, 1);
}

void State_Process::Reset(){
  ROS_WARN("Reset Process");
  time_last_scan = -1;
  b_first_frame = true;

}

void State_Process::set_gyr_cov_scale(const V3D &scaler) {
  cov_gyr_scale = scaler;
}

void State_Process::set_acc_cov_scale(const V3D &scaler) {
  cov_acc_scale = scaler;
}


State_Process::~State_Process(){}

void State_Process::only_propag(const RTVPointCloud::Ptr &meas, const double &timestamp, StatesGroup &state_input)
{
    cov_acc = Eye3d * cov_acc_scale;
    cov_gyr = Eye3d * cov_gyr_scale;
    
    const double &pcl_beg_time = timestamp;
    const double &pcl_end_time = timestamp + meas->back().time;

    MD(DIM_STATE,DIM_STATE) F_x,  cov_w;
    double dt = 0;

    if(b_first_frame){
        dt = 0.1;
        b_first_frame = false;
        time_last_scan = pcl_beg_time;
    }else{
        dt = pcl_beg_time - time_last_scan;
        time_last_scan = pcl_beg_time;
    }

    
    M3D Exp_f = Exp(state_input.bias_g,dt);

    F_x.setIdentity();
    cov_w.setZero();
    
    F_x.block<3, 3>(0, 0) = Exp(state_input.bias_g, -dt);
    F_x.block<3, 3>(0, 9) = Eye3d * dt;
    F_x.block<3, 3>(3, 6) = Eye3d * dt;
    cov_w.block<3, 3>(9, 9).diagonal() = cov_gyr * dt * dt; // for omega in constant model
    cov_w.block<3, 3>(6, 6).diagonal() = cov_acc * dt * dt; // for velocity in constant model

    state_input.cov = F_x * state_input.cov * F_x.transpose() + cov_w;
    state_input.rot = state_input.rot * Exp_f;
    state_input.pos = state_input.pos + state_input.vel * dt; 
}

