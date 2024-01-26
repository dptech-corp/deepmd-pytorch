// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPot.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace deepmd;

template <typename VALUETYPE>
std::vector<VALUETYPE> read_numbers(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::vector<VALUETYPE> numbers;
  std::string line;
  while (std::getline(file, line)) {
    try {
      VALUETYPE num = std::stod(line);
      numbers.push_back(num);
    } catch (const std::exception &e) {
      std::cerr << "Failed to parse number: " << line << ". Skipping..."
                << std::endl;
    }
  }

  return numbers;
}

int main() {
  std::vector<double> coord = read_numbers<double>("test_coord/coord0.txt");
  std::vector<int> atype = read_numbers<int>("test_coord/type.raw");
  std::vector<double> box = read_numbers<double>("test_coord/box0.txt");

  DeepPot dp;
  dp.init<double>("./frozen_model.pth");
  double energy;
  std::vector<double> force, virial;
  dp.compute<double, double>(energy, force, virial, coord, atype, box);

  printf("energy: %.6f\n", energy);
  printf("force:\n");
  for (int ii = 0; ii < atype.size(); ii++) {
    printf("%.6f %.6f %.6f\n", force[3 * ii + 0], force[3 * ii + 1],
           force[3 * ii + 2]);
  }
  printf("virial:\n");
  for (int ii = 0; ii < 3; ii++) {
    for (int jj = 0; jj < 3; jj++) {
      printf("%.6f ", virial[3 * ii + jj]);
    }
    printf("\n");
  }
  return 0;
}
