#pragma once

#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
#define PI_M (3.14159265358)
#define G_m_s2 (9.81)   // Gravaty const in GuangDong/China
#define DIM_STATE (15)  // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12) // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)
// old init
#define INIT_COV (0.0000001)
#define NUM_MATCH_POINTS (5)
#define MAX_MEAS_DIM (10000)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define CONSTRAIN(v, min, max) ((v > min) ? ((v < max) ? v : max) : min)
#define ARRAY_FROM_EIGEN(mat) mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)                                                \
  vector<decltype(mat)::Scalar>(mat.data(),                                    \
                                mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;


#define MD(a, b) Matrix<double, (a), (b)>
#define VD(a) Matrix<double, (a), 1>
#define MF(a, b) Matrix<float, (a), (b)>
#define VF(a) Matrix<float, (a), 1>

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);






namespace Eigen{



template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V>
using aligned_map = std::map<K, V, std::less<K>,
                             Eigen::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using aligned_unordered_map =
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                       Eigen::aligned_allocator<std::pair<K const, V>>>;

inline Eigen::Affine3d getTransBetween(Eigen::Vector3d trans_start,
                                       Eigen::Quaterniond rot_start,
                                       Eigen::Vector3d trans_end,
                                       Eigen::Quaterniond rot_end) {
  Eigen::Translation3d t_s(trans_start(0), trans_start(1), trans_start(2));
  Eigen::Translation3d t_e(trans_end(0), trans_end(1), trans_end(2));

  Eigen::Affine3d start = t_s * rot_start.toRotationMatrix();
  Eigen::Affine3d end = t_e * rot_end.toRotationMatrix();

  Eigen::Affine3d result = start.inverse() * end;
  return result;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> SkewSymmetric(const Eigen::Matrix<T, 3, 1>& w) {
  Eigen::Matrix<T, 3, 3> w_x;
  w_x << T(0), -w(2), w(1), w(2), T(0), -w(0), -w(1), w(0), T(0);
  return w_x;
}

/** sorts vectors from large to small
 * vec: vector to be sorted
 * sorted_vec: sorted results
 * ind: the position of each element in the sort result in the original vector
 * https://www.programmersought.com/article/343692646/
 */
inline void sort_vec(const Eigen::Vector3d& vec, Eigen::Vector3d& sorted_vec,
                     Eigen::Vector3i& ind) {
  ind = Eigen::Vector3i::LinSpaced(vec.size(), 0, vec.size() - 1);  //[0 1 2]
  auto rule = [vec](int i, int j) -> bool {
    return vec(i) > vec(j);
  };  // regular expression, as a predicate of sort

  std::sort(ind.data(), ind.data() + ind.size(), rule);

  // The data member function returns a pointer to the first element of
  // VectorXd, similar to begin()
  for (int i = 0; i < vec.size(); i++) {
    sorted_vec(i) = vec(ind(i));
  }
}

inline Eigen::Vector3d R2ypr(const Eigen::Matrix3d& R) {
  Eigen::Vector3d n = R.col(0);
  Eigen::Vector3d o = R.col(1);
  Eigen::Vector3d a = R.col(2);

  Eigen::Vector3d ypr(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * std::cos(y) + n(1) * std::sin(y));
  double r = atan2(a(0) * std::sin(y) - a(1) * std::cos(y),
                   -o(0) * std::sin(y) + o(1) * std::cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / M_PI * 180.0;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(
    const Eigen::MatrixBase<Derived>& ypr) {
  typedef typename Derived::Scalar Scalar_t;

  Scalar_t y = ypr(0) / 180.0 * M_PI;
  Scalar_t p = ypr(1) / 180.0 * M_PI;
  Scalar_t r = ypr(2) / 180.0 * M_PI;

  Eigen::Matrix<Scalar_t, 3, 3> Rz;
  Rz << std::cos(y), -std::sin(y), 0, std::sin(y), std::cos(y), 0, 0, 0, 1;

  Eigen::Matrix<Scalar_t, 3, 3> Ry;
  Ry << std::cos(p), 0., std::sin(p), 0., 1., 0., -std::sin(p), 0., std::cos(p);

  Eigen::Matrix<Scalar_t, 3, 3> Rx;
  Rx << 1., 0., 0., 0., std::cos(r), -std::sin(r), 0., std::sin(r), std::cos(r);

  return Rz * Ry * Rx;
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> positify(
    const Eigen::QuaternionBase<Derived>& q) {
  // printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
  // Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(),
  // -q.z()); printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z()); return
  // q.template w() >= (typename Derived::Scalar)(0.0) ? q :
  // Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
  return q;
}    
}