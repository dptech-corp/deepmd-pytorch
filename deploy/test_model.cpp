#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  auto device = torch::kCUDA;
  //auto device = torch::kCPU;
  torch::jit::script::Module module;
  // Deserialize the ScriptModule from a file using torch::jit::load().
  module = torch::jit::load(argv[1]);
  // Move the model to GPU.
  module.to(device);
  auto options = torch::TensorOptions();
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);
  // Create coord, atype, natoms, mapping, shift, selected 
  double coord_value[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  at::Tensor coord = torch::from_blob(coord_value, {1, 3, 3}, options).to(device);
  long atype_value[3] = {0, 1, 1};
  at::Tensor atype = torch::from_blob(atype_value, {1, 3}, int_options).to(device);
  long natoms_value[4] = {3, 3, 1, 2};
  at::Tensor natoms = torch::from_blob(natoms_value, {1, 4}, int_options).to(device);
  long mapping_value[3] = {0, 1, 2};
  at::Tensor mapping = torch::from_blob(mapping_value, {1, 3}, int_options).to(device);
  at::Tensor shift = at::zeros({1, 3, 3}, options).to(device);
  long selected_value[3*138] = {};
  for (int i=0; i<3; i++)
  {
    for (int j=0; j<138; j++)
    {
      selected_value[i*138+j] = -1;
    }
  }
  selected_value[0] = 1;
  selected_value[1] = 2;
  selected_value[138] = 0;
  selected_value[138*2] = 0;
  at::Tensor selected = torch::from_blob(selected_value, {1, 3, 138}, int_options).to(device);
  double box_value[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  at::Tensor box = torch::from_blob(box_value, {1, 3, 3}, options).to(device);
  // at::Tensor box = torch::randn({1, 3, 3}, options).to(device);
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(coord);
  inputs.push_back(atype);
  inputs.push_back(natoms);
  inputs.push_back(mapping);
  inputs.push_back(shift);
  inputs.push_back(selected);
  inputs.push_back(box); //selected_type, unused for se_a
  inputs.push_back(box); //selected_loc, unused for se_a
  inputs.push_back(box);
  c10::Dict<c10::IValue, c10::IValue> outputs = module.forward(inputs).toGenericDict();
  c10::IValue energy = outputs.at("energy");
  c10::IValue force = outputs.at("force");
  c10::IValue virial = outputs.at("virial");
  std::cout << energy<< force<< virial<<"ok\n";
}