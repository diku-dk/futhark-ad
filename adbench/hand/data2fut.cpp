// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <vector>

#include "light_matrix.h"

typedef struct
{
  int verts[3];
} Triangle;

class HandModelLightMatrix
{
public:
  std::vector<std::string> bone_names;
  std::vector<int> parents; // assumimng that parent is earlier in the order of bones
  std::vector<LightMatrix<double>> base_relatives;
  std::vector<LightMatrix<double>> inverse_base_absolutes;
  LightMatrix<double> base_positions;
  LightMatrix<double> weights;
  std::vector<Triangle> triangles;
  bool is_mirrored;
};

class HandDataLightMatrix
{
public:
  HandModelLightMatrix model;
  std::vector<int> correspondences;
  LightMatrix<double> points;
};

struct HandInput {
  std::vector<double> theta;
  HandDataLightMatrix data;
  std::vector<double> us;
};

void read_hand_model(const std::string& path, HandModelLightMatrix* pmodel)
{
  const char DELIMITER = ':';
  auto& model = *pmodel;
  std::ifstream bones_in(path + "bones.txt");
  std::string s;
  while (bones_in.good()) {
    getline(bones_in, s, DELIMITER);
    if (s.empty())
      continue;
    model.bone_names.push_back(s);
    getline(bones_in, s, DELIMITER);
    model.parents.push_back(std::stoi(s));
    double tmp[16];
    for (int i = 0; i < 16; i++) {
      getline(bones_in, s, DELIMITER);
      tmp[i] = std::stod(s);
    }
    model.base_relatives.emplace_back(4, 4);
    model.base_relatives.back().set(tmp);
    model.base_relatives.back().transpose_in_place();
    for (int i = 0; i < 15; i++) {
      getline(bones_in, s, DELIMITER);
      tmp[i] = std::stod(s);
    }
    getline(bones_in, s, '\n');
    tmp[15] = std::stod(s);
    model.inverse_base_absolutes.emplace_back(4, 4);
    model.inverse_base_absolutes.back().set(tmp);
    model.inverse_base_absolutes.back().transpose_in_place();
  }
  bones_in.close();
  int n_bones = (int)model.bone_names.size();

  std::ifstream vert_in(path + "vertices.txt");
  int n_vertices = 0;
  while (vert_in.good()) {
    getline(vert_in, s);
    if (!s.empty())
      n_vertices++;
  }
  vert_in.close();

  model.base_positions.resize(4, n_vertices);
  model.base_positions.set_row(3, 1.);
  model.weights.resize(n_bones, n_vertices);
  model.weights.fill(0.);
  vert_in = std::ifstream(path + "vertices.txt");
  for (int i_vert = 0; i_vert < n_vertices; i_vert++) {
    for (int j = 0; j < 3; j++) {
      getline(vert_in, s, DELIMITER);
      model.base_positions(j, i_vert) = std::stod(s);
    }
    for (int j = 0; j < 3 + 2; j++) {
      getline(vert_in, s, DELIMITER); // skip
    }
    getline(vert_in, s, DELIMITER);
    int n = std::stoi(s);
    for (int j = 0; j < n; j++) {
      getline(vert_in, s, DELIMITER);
      int i_bone = std::stoi(s);
      if (j == n - 1)
        getline(vert_in, s, '\n');
      else
        getline(vert_in, s, DELIMITER);
      model.weights(i_bone, i_vert) = std::stod(s);
    }
  }
  vert_in.close();

  std::ifstream triangles_in(path + "triangles.txt");
  std::string ss[3];
  while (triangles_in.good()) {
    getline(triangles_in, ss[0], DELIMITER);
    if (ss[0].empty())
      continue;

    getline(triangles_in, ss[1], DELIMITER);
    getline(triangles_in, ss[2], '\n');
    Triangle curr;
    for (int i = 0; i < 3; i++)
      curr.verts[i] = std::stoi(ss[i]);
    model.triangles.push_back(curr);
  }
  triangles_in.close();

  model.is_mirrored = false;
}

void read_hand_instance(const std::string& model_dir,
                        const std::string& fn_in,
                        std::vector<double>* theta,
                        HandDataLightMatrix* data,
                        std::vector<double>* us)
{
  read_hand_model(model_dir, &data->model);
  std::ifstream in(fn_in);
  int n_pts, n_theta;
  in >> n_pts >> n_theta;
  data->correspondences.resize(n_pts);
  data->points.resize(3, n_pts);
  for (int i = 0; i < n_pts; i++) {
    in >> data->correspondences[i];
    for (int j = 0; j < 3; j++) {
      in >> data->points(j, i);
    }
  }
  if (us != nullptr) {
    us->resize(2 * n_pts);
    for (int i = 0; i < 2 * n_pts; i++) {
      in >> (*us)[i];
    }
  }
  theta->resize(n_theta);
  for (int i = 0; i < n_theta; i++) {
    in >> (*theta)[i];
  }
  in.close();
}

void print_fut(const int& x) {
  std::cout << x << "i32";
}


void print_fut(const double& x) {
  std::cout << x << "f64";
}

void print_fut(const bool& x) {
  std::cout << (x ? "true" : "false");
}

void print_fut(const Triangle& t) {
  std::cout << "[";
  print_fut(t.verts[0]);
  std::cout << ", ";
  print_fut(t.verts[1]);
  std::cout << ", ";
  print_fut(t.verts[2]);
  std::cout << "]" << std::endl;
}

void print_fut(const LightMatrix<double> &m) {
  std::cout << "[";
  for (int i = 0; i < m.rows(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "[";
    for (int j = 0; j < m.cols(); j++) {
      if (j > 0) {
        std::cout << ", ";
      }
      print_fut(m(i,j));
    }
    std::cout << "]";
  }
  std::cout << "]";
}

template<typename T>
void print_vec(const std::vector<T> &v) {
  bool first = true;
  std::cout << "[";
  for (auto x : v) {
    if (!first) {
      std::cout << ", ";
    }
    print_fut(x);
    first = false;
  }
  std::cout << "]";
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " " << "<input_file> <model_dir>" << std::endl;
    return 1;
  }

  const char *input_file = argv[1];
  const char *model_dir = argv[2];

  HandInput input;

  read_hand_instance(model_dir, input_file, &input.theta, &input.data, nullptr);

  print_vec(input.data.model.parents); std::cout << std::endl;
  print_vec(input.data.model.base_relatives); std::cout << std::endl;
  print_vec(input.data.model.inverse_base_absolutes); std::cout << std::endl;
  print_fut(input.data.model.weights); std::cout << std::endl;
  print_fut(input.data.model.base_positions); std::cout << std::endl;
  print_vec(input.data.model.triangles); std::cout << std::endl;
  print_fut(input.data.model.is_mirrored); std::cout << std::endl;

  print_vec(input.data.correspondences); std::cout << std::endl;
  print_fut(input.data.points); std::cout << std::endl;
  print_vec(input.theta); std::cout << std::endl;
}
