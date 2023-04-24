#include <glog/logging.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf2_msgs/TFMessage.h>
#include "omp.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <cmath>
#include <iostream> 
#include "../utils/mypcl_cloud_type.h"
#include "../odometry/map.hpp"
#include "../odometry/state.hpp"
#define HASH_P 116101
#define MAX_N 10000000000

using namespace std;
using namespace Eigen;

string lidar_topic;
double leaf_size = 0.2;
double range_threshold = 100;
double range_cov = 0.02;
double angle_cov = 0.05;
double voxel_size = 1.0;
double min_eigen_value = 0.1;
double res_mean_last = 0.5;
double gyr_cov_scale, acc_cov_scale;
int point_filter_num = 1;
int max_layer = 3;
int NUM_MAX_ITERATIONS = 3;
int max_points_size = 100;
int max_cov_points_size = 100;
std::vector<int> layer_size;

vector<double> layer_point_size;
int rematch_num = 0;

bool  flg_EKF_converged,init_map = false,EKF_stop_flag = 0,flg_EKF_inited = 0;
bool dense_map_enable;
int pub_point_cloud_skip;
std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
geometry_msgs::Quaternion geoQuat;
// record time
double undistort_time_mean = 0;
double down_sample_time_mean = 0;
double calc_cov_time_mean = 0;
double scan_match_time_mean = 0;
double ekf_solve_time_mean = 0;
double map_update_time_mean = 0;

double map_incremental_time, kdtree_search_time, total_time, scan_match_time,
    solve_time;

int iterCount,frameCount = 0;

VD(DIM_STATE) solution;
MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
V3D rot_add,t_add,v_add;
StatesGroup state_propagat;
double deltaT, deltaR, aver_time_consu = 0;
V3D euler_cur;
V3D position_last(Zero3d);
shared_ptr<State_Process> process_iekf(new State_Process());
ros::Publisher pubOdometry,pubPath,pubLaserCloudFullRes;
nav_msgs::Path path;
void sub_sample(RTVPointCloud::Ptr &input_points,double size_voxel,double range)
{   
    unordered_map<VOXEL_LOC,vector<RTVPoint>> grid_map;
    for(uint i = 0; i < input_points->size(); i++)
    {   
        if(sqrt(input_points->points[i].x * input_points->points[i].x + input_points->points[i].y * input_points->points[i].y 
        +input_points->points[i].z * input_points->points[i].z) < 150)
        {   
            float loc_xyz[3];
            loc_xyz[0] = input_points->points[i].x / size_voxel;
            loc_xyz[1] = input_points->points[i].y / size_voxel;
            loc_xyz[2] = input_points->points[i].z / size_voxel;
            for(int j = 0 ; j < 3 ; j ++)
            {
                if(loc_xyz[j] < 0)
                {
                    loc_xyz[j] -= 1.0;
                }
            }
            VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            grid_map[position].push_back(input_points->points[i]);
        }
        
    }

    input_points->resize(0);
    for(const auto & n : grid_map)
    {
        if(n.second.size() > 0)
        {
            input_points->points.push_back(n.second[0]);
        }
    }

}

StatesGroup state;
vector<StatesGroup> state_list;
RTVPointCloud::Ptr normvec(new RTVPointCloud(100000, 1));
RTVPointCloud::Ptr laserCloudOri(new RTVPointCloud(100000, 1));
RTVPointCloud::Ptr laserCloudNoeffect(new RTVPointCloud(100000, 1));
RTVPointCloud::Ptr corr_normvect(new RTVPointCloud(100000, 1));

const bool time_list(RTVPoint &x, RTVPoint &y) {
  return (x.time < y.time);
};
const bool var_contrast(pointWithCov &x, pointWithCov &y) {
  return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};
void pointBodyToWorld(RTVPoint const *const pi, RTVPoint *const po){
V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state.rot * (p_body) + state.pos);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->time = pi->time;
  po->velocity = pi->velocity;
}

void transformLidar(const StatesGroup &state,const RTVPointCloud::Ptr &input_cloud,RTVPointCloud::Ptr &trans_cloud){
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
    // RTVPoint p_c = input_cloud->points[i];    
    // if(input_cloud->points[i].x ==0 && input_cloud->points[i].y ==0 &&input_cloud->points[i].z ==0)
    // {
    //   std::cout<<"0000000000000000"<<std::endl;
    // }
    Eigen::Vector3d p;
    p << input_cloud->points[i].x,input_cloud->points[i].y,input_cloud->points[i].z;
    Eigen::Vector3d po;
    po = state.rot * p + state.pos;
    RTVPoint pi;
    pi.x = po(0);
    pi.y = po(1);
    pi.z = po(2);

    pi.velocity = input_cloud->points[i].velocity;
    pi.time = input_cloud->points[i].time;
    // if(pi.x ==0 && pi.y ==0 && pi.z ==0)
    // {
    //   std::cout<<"11111111111111111"<<std::endl;
    // }
    trans_cloud->points.push_back(pi);
  }
}

void pclCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg){
    std_msgs::Header current_header = msg->header;
    
    RTVPointCloud::Ptr ptr(new RTVPointCloud());
    pcl::fromROSMsg(*msg,*ptr);

    /*downsample*/
    cout<<"raw_points.size()"<<ptr->points.size()<<endl;
    auto undistort_start = std::chrono::high_resolution_clock::now();
    // auto t_downsample_start = std::chrono::high_resolution_clock::now();
    // sub_sample(ptr,leaf_size,range_threshold);
    // auto t_downsample_end = std::chrono::high_resolution_clock::now();
    // auto t_downsample = std::chrono::duration_cast<std::chrono::duration<double>>(t_downsample_end - t_downsample_start).count() * 1000;

    RTVPointCloud::Ptr undistort_cloud(new RTVPointCloud());
    // undistort_cloud->clear();
    // undistort_cloud->resize(ptr->size());
    // undistort_cloud->is_dense = ptr->is_dense;
    /*undistort*/
    int i = 0;
    for(auto point : ptr->points)
    {   
        if(pcl_isnan(point.x)|| pcl_isnan(point.y) || pcl_isnan(point.z)  || point.x == 0 || point.y == 0 || point.z == 0)
        continue;
        if(i%point_filter_num == 0)
        {
            double dt = point.time;
            double radial_velocity = point.velocity;
            Vector3d point3d(point.x,point.y,point.z);

            double normal = std::sqrt(point3d.transpose() * point3d);
            double velocity_x = (radial_velocity * point3d.x()) / normal;
            double velocity_y = (radial_velocity * point3d.y()) / normal;
            double velocity_z = (radial_velocity * point3d.z()) / normal;
            Eigen::Vector3d relvative_translation = Eigen::Vector3d((-velocity_x*dt),(-velocity_y*dt),(-velocity_z*dt));
            Eigen::Vector3d new_point = point3d  + relvative_translation;
            RTVPoint new_point_in_cloud;

            new_point_in_cloud.x = new_point.x();
            new_point_in_cloud.y = new_point.y();
            new_point_in_cloud.z = new_point.z();
            new_point_in_cloud.time = point.time;
            new_point_in_cloud.velocity = radial_velocity;
            undistort_cloud->points.push_back(new_point_in_cloud);
        }
        i++;
    }
    cout<<"undistort_cloud.size():"<<undistort_cloud->size()<<endl;
    auto undistort_end = std::chrono::high_resolution_clock::now();
    auto undistort_time = std::chrono::duration_cast<std::chrono::duration<double>>(undistort_end - undistort_start).count() * 1000;
    cout<<"undistort_time="<<undistort_time<<std::endl;



    if(!init_map)
    {
        process_iekf->Reset();
    }

    process_iekf->only_propag(undistort_cloud,current_header.stamp.toSec(),state);


    if(!init_map)
    {   cout<<"voxelMap inited!!!!"<<endl;
        RTVPointCloud::Ptr world_lidar(new RTVPointCloud());
        transformLidar(state,undistort_cloud,world_lidar);
        vector<pointWithCov> pv_list;
        for(size_t i = 0; i < world_lidar->size(); i ++)
        {   
            pointWithCov pv;
            pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;

            Eigen::Vector3d point_this(undistort_cloud->points[i].x, undistort_cloud->points[i].y,undistort_cloud->points[i].z);
            if(point_this[2] == 0)
            {
                point_this[2] = 0.001;
            }

            Eigen::Matrix3d cov;
            calcBodyCov(point_this, range_cov, angle_cov, cov);
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_this);
            cov = state.rot * cov * state.rot.transpose() + (-point_crossmat)*state.cov.block<3,3>(0,0)*
            (-point_crossmat).transpose() + state.cov.block<3,3>(3,3);

            pv.cov = cov;
            pv_list.push_back(pv);
            Eigen::Vector3d sigma_pv = pv.cov.diagonal();
            sigma_pv[0] = sqrt(sigma_pv[0]);
            sigma_pv[1] = sqrt(sigma_pv[1]);
            sigma_pv[2] = sqrt(sigma_pv[2]);

        }

        buildVoxelMap(pv_list,voxel_size,max_layer,layer_size,max_points_size,
        max_points_size,min_eigen_value,voxel_map);
        init_map = true;
        return;
    }
    
    cout<<"frame count:"<<frameCount<<endl;
    
    auto t_downsample_start = std::chrono::high_resolution_clock::now();
    sub_sample(undistort_cloud,leaf_size,range_threshold);
    auto t_downsample_end = std::chrono::high_resolution_clock::now();
    auto t_downsample = std::chrono::duration_cast<std::chrono::duration<double>>(t_downsample_end - t_downsample_start).count() * 1000;    

    sort(undistort_cloud->points.begin(),undistort_cloud->points.end(),time_list);
    
    int rematch = 0;

    bool nearest_search_en = true;

    double total_residual;

    scan_match_time = 0.0;

    vector<M3D> body_var;
    vector<M3D> crossmat_list;

    /*** iterated state estimation ***/
    auto calc_point_cov_start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < undistort_cloud->size(); i++){
        V3D point_this(undistort_cloud->points[i].x,
        undistort_cloud->points[i].y,undistort_cloud->points[i].z);

        if (point_this[2] == 0) {
          point_this[2] = 0.001;
        }

        M3D cov;
        calcBodyCov(point_this,range_cov,angle_cov,cov);
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);
        crossmat_list.push_back(point_crossmat);
        M3D rot_var = state.cov.block<3, 3>(0, 0);
        M3D t_var = state.cov.block<3, 3>(3, 3);
        body_var.push_back(cov);
    }
    auto calc_point_cov_end = std::chrono::high_resolution_clock::now();
    double calc_point_cov_time = std::chrono::duration_cast<std::chrono::duration<double>>(calc_point_cov_end - calc_point_cov_start).count() * 1000;
    state_propagat = state;
    cout<<"pos:"<<state.pos.transpose()<<endl;
    cout<<"rot:"<<state.rot<<endl;
    for(iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++){
        laserCloudOri->clear();
        laserCloudNoeffect->clear();
        corr_normvect->clear();
        total_residual = 0.0;

        vector<double> r_list;
        vector<ptpl> ptpl_list;
        /** LiDAR match based on 3 sigma criterion **/

        vector<pointWithCov> pv_list;
        vector<M3D> var_list;
        RTVPointCloud::Ptr world_lidar(new RTVPointCloud());
        transformLidar(state,undistort_cloud,world_lidar);
        for(size_t i = 0; i < undistort_cloud->size() ; i++){
            pointWithCov pv;
            pv.point << undistort_cloud->points[i].x,undistort_cloud->points[i].y,undistort_cloud->points[i].z;
            pv.point_world << world_lidar->points[i].x,world_lidar->points[i].y,world_lidar->points[i].z;
            pv.radial_vel = undistort_cloud->points[i].velocity;
            pv.time = undistort_cloud->points[i].time;
            M3D cov = body_var[i];
            M3D point_crossmat = crossmat_list[i];
            M3D rot_var = state.cov.block<3, 3>(0, 0);
            M3D t_var = state.cov.block<3, 3>(3, 3);
            cov = state.rot * cov * state.rot.transpose() + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;

            pv.cov = cov;
            pv_list.push_back(pv);
            var_list.push_back(cov);
        }
        auto scan_match_time_start = std::chrono::high_resolution_clock::now();
        std::vector<V3D> non_match_list;

        BuildResidualListOMP(voxel_map, voxel_size, 3.0, max_layer, pv_list, ptpl_list, non_match_list);
        auto scan_match_time_end = std::chrono::high_resolution_clock::now();

        int effect_feat_num = 0;
        // double sum_radial=0;
        // double sum_radial_vel = 0;
        for(int i = 0; i < ptpl_list.size();i++)
        {

            // Eigen::Vector3d v_body = state.rot.inverse() * state.vel;
            // double radial;
            // radial =  (ptpl_list[i].point.transpose() * v_body);
            // double norm =  sqrt(ptpl_list[i].point.transpose() * ptpl_list[i].point);
            // radial /= norm;
            
            // sum_radial+=radial;
            // sum_radial_vel+=ptpl_list[i].radial_vel;
            RTVPoint pi_body;
            RTVPoint pi_world;
            RTVPoint pl;
            pi_body.x = ptpl_list[i].point(0);
            pi_body.y = ptpl_list[i].point(1);
            pi_body.z = ptpl_list[i].point(2);
            pi_body.velocity = ptpl_list[i].radial_vel;
            pointBodyToWorld(&pi_body, &pi_world);
            pl.x = ptpl_list[i].normal(0);
            pl.y = ptpl_list[i].normal(1);
            pl.z = ptpl_list[i].normal(2);
            effect_feat_num++;
            float dis = (pi_world.x * pl.x + pi_world.y * pl.y +
            pi_world.z * pl.z + ptpl_list[i].d);
            pl.intensity = dis;

            laserCloudOri->push_back(pi_body);
            corr_normvect->push_back(pl);
            total_residual += fabs(dis);
        }
        res_mean_last = total_residual / effect_feat_num;
        scan_match_time += std::chrono::duration_cast<std::chrono::duration<double>>(scan_match_time_end - scan_match_time_start).count() * 1000;
        auto t_solve_start = std::chrono::high_resolution_clock::now();
        // sum_radial/=effect_feat_num;
        // sum_radial_vel/=effect_feat_num;
        // cout<<"sum_radial:"<<sum_radial<<";sum_radial_vel:"<<sum_radial_vel<<endl;
        /*** Computation of Measuremnt Jacobian matrix H and measurents vector
         * ***/
        MatrixXd Hsub(effect_feat_num*2,DIM_STATE);
        MatrixXd Hsub_T_R_inv(DIM_STATE,effect_feat_num*2);
        VectorXd R_inv(effect_feat_num);
        VectorXd meas_vec(effect_feat_num*2);

        // MatrixXd Hsub(effect_feat_num, 6);
        // MatrixXd Hsub_T_R_inv(6, effect_feat_num);
        // VectorXd R_inv(effect_feat_num);
        // VectorXd meas_vec(effect_feat_num);

        Hsub.setZero();
        Hsub_T_R_inv.setZero();
        R_inv.setZero();
        meas_vec.setZero();

        for(int i = 0; i < effect_feat_num; i++)
        {
            const RTVPoint &laser_p = laserCloudOri->points[i];
            V3D point_this(laser_p.x,laser_p.y,laser_p.z);
            M3D cov;
            calcBodyCov(point_this, range_cov, angle_cov, cov);

            cov = state.rot * cov * state.rot.transpose();

            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_this);

            const RTVPoint &norm_p = corr_normvect->points[i];
            V3D norm_vec(norm_p.x,norm_p.y,norm_p.z);
            V3D point_world = state.rot * point_this + state.pos;

            // /*** get the normal vector of closest surface/corner ***/
            Eigen::Matrix<double,1,6> J_nq;
            J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
            J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
            double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();
            R_inv(i) = 1.0 / (sigma_l + norm_vec.transpose() * cov * norm_vec);
            double ranging_dis = point_this.norm();
            /*** calculate the Measuremnt Jacobian matrix H ***/
            /**点到平面残差*/
            V3D A(point_crossmat * state.rot.transpose() * norm_vec);
            // Hsub.row(i)<<VEC_FROM_ARRAY(A),norm_p.x,norm_p.y,norm_p.z;
            // cout<<"R_inv(i):"<<R_inv(i)<<endl;
            // cout<<"meas_vec(i):"<<-norm_p.intensity;
            Hsub.block<1,6>(i,0)<<VEC_FROM_ARRAY(A),norm_p.x,norm_p.y,norm_p.z;
            // Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i),
            //   A[2] * R_inv(i), norm_p.x * R_inv(i), norm_p.y * R_inv(i),
            //   norm_p.z * R_inv(i);

            Hsub_T_R_inv.block<6,1>(0,i) << A[0] * R_inv(i), A[1] * R_inv(i),
              A[2] * R_inv(i), norm_p.x * R_inv(i), norm_p.y * R_inv(i),
              norm_p.z * R_inv(i);
            meas_vec(i) = -norm_p.intensity;
            /*多普勒速度残差*/
            double denominator = sqrt(point_this.transpose() * point_this);

            V3D vel_body = state.rot.inverse() * state.vel;
            M3D skew_vel_body ;
            skew_vel_body << SKEW_SYM_MATRX(vel_body);
            V3D dr_dR = - point_this.transpose() * skew_vel_body / denominator;

            V3D dr_dv = -point_this.transpose() * state.rot.transpose() / denominator;
            double body_vel = point_this.transpose() * state.rot.inverse() * state.vel;// / denominator;
            body_vel /= denominator;
            // double velo_esti = point_this.transpose() * state.rot.transpose() * state.vel ;/// denominator;
            // velo_esti /= denominator;
            Hsub.block<1,3>(i+effect_feat_num,0) << VEC_FROM_ARRAY(dr_dR);
            Hsub.block<1,3>(i+effect_feat_num,6) << VEC_FROM_ARRAY(dr_dv);
            Hsub_T_R_inv.block<3,1>(0,i+effect_feat_num)<<dr_dR[0],dr_dR[1],dr_dR[2];
            Hsub_T_R_inv.block<3,1>(6,i+effect_feat_num)<<dr_dv[0],dr_dv[1],dr_dv[2];
            meas_vec(i+effect_feat_num) = -(laser_p.velocity - body_vel);
            // meas_vec(i+effect_feat_num) = -( body_vel - laser_p.velocity);

        }

        // MatrixXd K(DIM_STATE,effect_feat_num);
        MatrixXd K(DIM_STATE,2*effect_feat_num);

        EKF_stop_flag = false;
        flg_EKF_converged = false;

        /*** Iterative Kalman Filter Update ***/
        if(!flg_EKF_inited){
            flg_EKF_inited = true;
            cout << "||||||||||Initiallizing LiDar||||||||||" << endl;
            /*** only run in initialization period ***/
            MatrixXd H_init(MD(9,DIM_STATE)::Zero());
            MatrixXd z_init(VD(9)::Zero());
            H_init.block<3,3>(0,0) = M3D::Identity();
            H_init.block<3,3>(3,3) = M3D::Identity();

            z_init.block<3,1>(0,0) = -Log(state.rot);

            z_init.block<3,1>(3,0) = -state.pos;

            auto H_init_T = H_init.transpose();

            auto &&K_init = state.cov * H_init_T *  (H_init * state.cov * H_init_T + 0.0001 * MD(9, 9)::Identity()).inverse();            
            solution = K_init * z_init;
            // cout<<"solution:"<<solution.transpose()<<endl;
            state.resetpose();
            EKF_stop_flag = true;
        }else{
            cout<<"|||||||||||||||||||KF_update||||||||||||||||||||||||"<<endl;

            auto &&Hsub_T = Hsub.transpose();

            H_T_H = Hsub_T_R_inv * Hsub;
            // H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
            // cout<<"HTH:"<<H_T_H.block<6, 6>(0, 0)<<endl;
            // cout<<"state.cov.inverse():"<<(state.cov).inverse()<<endl;

            MD(DIM_STATE,DIM_STATE) &&K_1 = (H_T_H + (state.cov).inverse()).inverse();
            K = K_1 * Hsub_T_R_inv;
            // cout<<"K_1:"<<K_1.block<DIM_STATE, 6>(0, 0)<<endl;

            // K = K_1.block<DIM_STATE, 6>(0, 0) * Hsub_T_R_inv;

            auto vec = state_propagat - state;

            // solution = K * meas_vec + vec - K * Hsub * vec.block<6, 1>(0, 0);
            solution = K * meas_vec + vec - K * Hsub * vec;
            
            cout<<"solution:"<<solution.transpose()<<endl;
            // solution = K * meas_vec + vec - K * Hsub * vec;
            // if(std::isnan(solution(10)))
            // {
            //     // ros::shutdown();
            //     std::cin.get();
            // }
            state += solution;
            rot_add = solution.block<3,1>(0,0);
            t_add = solution.block<3,1>(3,0);
            v_add = solution.block<3,1>(6,0);

            if((rot_add.norm() * 57.3) < 0.01 & (t_add.norm()) * 100 <0.015 ){
                flg_EKF_converged = true;
            }

            deltaR = rot_add.norm() * 57.3;
            deltaT = t_add.norm() * 100;
        }
        euler_cur = RotMtoEuler(state.rot);
        /*** Rematch Judgement ***/        
        nearest_search_en = false;
        if(flg_EKF_converged || (rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2))){
            nearest_search_en = true;
            rematch_num++;
        }
        /*** Convergence Judgements and Covariance Update ***/

        if(!EKF_stop_flag && (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))){
            if(flg_EKF_inited){
                /*** Covariance Update ***/
                G.setZero();
                // G.block<DIM_STATE, 6>(0, 0) = K * Hsub;

                G = K * Hsub;
                state.cov = (I_STATE - G) * state.cov;

                total_residual += (state.pos - position_last).norm();
                position_last = state.pos;
                geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0),euler_cur(1),euler_cur(2));

                VD(DIM_STATE) K_sum = K.rowwise().sum();
                VD(DIM_STATE) P_diag = state.cov.diagonal();
                 
            }
            EKF_stop_flag = true;
        }
        
        auto t_solve_end = std::chrono::high_resolution_clock::now();
        solve_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_solve_end - t_solve_start).count() * 1000;
        if (EKF_stop_flag)
          break;        
    }

    // auto last_state = state_list.back();
    // auto q_begin = Eigen::Quaterniond(last_state.rot);
    // auto q_end = Eigen::Quaterniond(state.rot);
    // V3D t_begin = last_state.pos;
    // V3D t_end = state.pos;
    // for(auto &point : undistort_cloud->points){
    //     double alpha_timestamp = point.time / 0.1;
    //     M3D R = q_begin.slerp(alpha_timestamp,q_end).normalized().toRotationMatrix();
    //     V3D t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
    //     V3D point_correct = R * V3D(point.x,point.y,point.z) + t;
    //     point.x = point_correct.x();
    //     point.y = point_correct.y();
    //     point.z = point_correct.z();
    // }

    // state_list.push_back(state);

    /*** add the  points to the voxel map ***/
    auto map_incremental_start = std::chrono::high_resolution_clock::now();
    RTVPointCloud::Ptr world_lidar(new RTVPointCloud());
    transformLidar(state,undistort_cloud,world_lidar);
    std::vector<pointWithCov> pv_list;
    for (size_t i = 0; i < world_lidar->size(); i++) {

        if(pcl_isnan(world_lidar->points[i].x))
        continue;
        pointWithCov pv;
        
        pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
            world_lidar->points[i].z;
        M3D point_crossmat = crossmat_list[i];
        M3D cov = body_var[i];
        cov = state.rot * cov * state.rot.transpose() +
                (-point_crossmat) * state.cov.block<3, 3>(0, 0) *
                    (-point_crossmat).transpose() +
                state.cov.block<3, 3>(3, 3);
        pv.cov = cov;
        pv_list.push_back(pv);
    }
    std::sort(pv_list.begin(), pv_list.end(), var_contrast);
    updateVoxelMap(pv_list, voxel_size, max_layer, layer_size, max_points_size, max_points_size, min_eigen_value, voxel_map);
    auto map_incremental_end = std::chrono::high_resolution_clock::now();
    map_incremental_time = std::chrono::duration_cast<std::chrono::duration<double>>(map_incremental_end - map_incremental_start).count() * 1000;

    total_time = t_downsample + scan_match_time + solve_time + map_incremental_time + undistort_time + calc_point_cov_time;
    frameCount++;
    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "world";
    odomAftMapped.child_frame_id = "lidar";
    odomAftMapped.header.stamp = current_header.stamp;
    odomAftMapped.pose.pose.position.x = state.pos(0);
    odomAftMapped.pose.pose.position.y = state.pos(1);
    odomAftMapped.pose.pose.position.z = state.pos(2);
    odomAftMapped.pose.pose.orientation.x = geoQuat.x;
    odomAftMapped.pose.pose.orientation.y = geoQuat.y;
    odomAftMapped.pose.pose.orientation.z = geoQuat.z;
    odomAftMapped.pose.pose.orientation.w = geoQuat.w;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(state.pos(0), state.pos(1), state.pos(2)));    
    q.setW(geoQuat.w);
    q.setX(geoQuat.x);
    q.setY(geoQuat.y);
    q.setZ(geoQuat.z);    
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "lidar"));
    pubOdometry.publish(odomAftMapped);

    geometry_msgs::PoseStamped body_pose;
    body_pose.header.stamp = current_header.stamp;
    body_pose.header.frame_id = "world";
    body_pose.pose.position.x = state.pos(0);
    body_pose.pose.position.y = state.pos(1);
    body_pose.pose.position.z = state.pos(2);
    body_pose.pose.orientation.x = geoQuat.x;
    body_pose.pose.orientation.y = geoQuat.y;
    body_pose.pose.orientation.z = geoQuat.z;
    body_pose.pose.orientation.w = geoQuat.w;

    path.header.stamp = current_header.stamp;
    path.header.frame_id = "world";
    path.poses.push_back(body_pose);
    pubPath.publish(path);
    transformLidar(state, undistort_cloud, world_lidar);
    sensor_msgs::PointCloud2 pub_cloud;
    pcl::toROSMsg(*world_lidar,pub_cloud);
    pub_cloud.header.stamp = current_header.stamp;
    pub_cloud.header.frame_id = "world";
    if(frameCount%pub_point_cloud_skip == 0)
    {
        pubLaserCloudFullRes.publish(pub_cloud);
    }


    

    undistort_time_mean = undistort_time_mean * (frameCount - 1) / frameCount +
                        (undistort_time) / frameCount;
    down_sample_time_mean =
        down_sample_time_mean * (frameCount - 1) / frameCount +
        (t_downsample) / frameCount;
    calc_cov_time_mean = calc_cov_time_mean * (frameCount - 1) / frameCount +
                        (calc_point_cov_time) / frameCount;
    scan_match_time_mean =
        scan_match_time_mean * (frameCount - 1) / frameCount +
        (scan_match_time) / frameCount;
    ekf_solve_time_mean = ekf_solve_time_mean * (frameCount - 1) / frameCount +
                        (solve_time) / frameCount;
    map_update_time_mean =
        map_update_time_mean * (frameCount - 1) / frameCount +
        (map_incremental_time) / frameCount;

    aver_time_consu = aver_time_consu * (frameCount - 1) / frameCount +
                    (total_time) / frameCount;

    cout << "[ Time ]: "
        << "average undistort: " << undistort_time_mean << std::endl;
    cout << "[ Time ]: "
        << "average down sample: " << down_sample_time_mean << std::endl;
    cout << "[ Time ]: "
        << "average calc cov: " << calc_cov_time_mean << std::endl;
    cout << "[ Time ]: "
        << "average scan match: " << scan_match_time_mean << std::endl;
    cout << "[ Time ]: "
        << "average solve: " << ekf_solve_time_mean << std::endl;
    cout << "[ Time ]: "
        << "average map incremental: " << map_update_time_mean << std::endl;
    cout << "[ Time ]: "
        << " average total " << aver_time_consu << endl;

}


int main(int argc,char **argv){
    ros::init(argc,argv,"4D_LO");
    ros::NodeHandle nh;
    nh.param<string>("common/lid_topic",lidar_topic,"/lidar_topic");
    
    nh.param<double>("mapping/leaf_size",leaf_size,0.2);
    nh.param<double>("mapping/range_threshold",range_threshold,100);
    nh.param<double>("noise_model/range_cov",range_cov,0.02);
    nh.param<double>("noise_model/angle_cov",angle_cov,0.05);
    nh.param<double>("noise_model/acc_cov_scale",acc_cov_scale,0.1);
    nh.param<double>("noise_model/gyr_cov_scale",gyr_cov_scale,0.1);
    nh.param<double>("mapping/voxel_size", voxel_size,1.0);
    nh.param<double>("mapping/plannar_threshold",min_eigen_value,0.1);
    nh.param<int>("mapping/max_layer",max_layer,3);
    nh.param<int>("mapping/max_iteration",NUM_MAX_ITERATIONS,3);
    nh.param<int>("mapping/max_points_size",max_points_size,100);
    nh.param<int>("mapping/max_cov_points_size",max_cov_points_size,100);
    nh.param<int>("mapping/point_filter_num",point_filter_num,1);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size,vector<double>());
    for (int i = 0; i < layer_point_size.size(); i++) {
        layer_size.push_back(layer_point_size[i]);
    }
    nh.param<bool>("visualization/dense_map_enable",dense_map_enable,true);
    nh.param<int>("visualization/pub_point_cloud_skip",pub_point_cloud_skip,5);
    ros::Subscriber sub_pcl = nh.subscribe(lidar_topic,2000,pclCallBack);
    pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered",100);
    pubOdometry = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init",10);
    pubPath = nh.advertise<nav_msgs::Path>("/path",10);

    process_iekf->set_acc_cov_scale(V3D(acc_cov_scale,acc_cov_scale,acc_cov_scale));
    process_iekf->set_gyr_cov_scale(V3D(gyr_cov_scale,gyr_cov_scale,gyr_cov_scale));
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
    ros::spin();

    return 0;
}