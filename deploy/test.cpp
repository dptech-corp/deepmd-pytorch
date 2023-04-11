#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  auto device = torch::kCUDA;
  //auto device = torch::kCPU;
  // Deserialize the ScriptModule from a file using torch::jit::load().
  module.to(device);
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  auto options = torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true);
  inputs.push_back(torch::ones({10, 3}, options).to(device));

  // Execute the model and turn its output into a tensor.
  auto outputs = module.forward(inputs).toTensorVector();
  at::Tensor energy = outputs[0];
  at::Tensor force = outputs[1];
  std::cout <<energy << force << "ok\n";
}