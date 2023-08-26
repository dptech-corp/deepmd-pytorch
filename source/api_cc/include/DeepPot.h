#include "common.h"

namespace deepmd {
/**
 * @brief Deep Potential.
 **/
class DeepPot {
 public:
  // cublasHandle_t handle;
  /**
   * @brief DP constructor without initialization.
   **/
  DeepPot();
  ~DeepPot();
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the frozen model file.
   **/
  template <typename VALUETYPE>
  void init(const std::string& model);

  int numb_types() { return sec.size(); }

  /**
   * @brief Evaluate the energy, force and virial by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box);
 private:
  torch::jit::script::Module module;
  double rcut;
  std::vector<int> sec;
};

template <typename VALUETYPE>
void DeepPot::init(const std::string& model) {
    // cublasCreate(&handle);

    try {
        module = torch::jit::load(model);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
    }


    auto rcut_ = module.run_method("get_rcut").toDouble();
    rcut = static_cast<VALUETYPE>(rcut_);

    auto sec_ = module.run_method("get_sec");
    TensortoVec<int64_t, int>(sec_, sec);
    /*
    std::vector<int64_t> values;
    if (sec_.isTensorList()) {
        auto tensor_list = sec_.toTensorList().vec();  // Extract the underlying vector
        for (const at::Tensor& tensor : tensor_list) {
            values.push_back(tensor.item<int64_t>());
        }
    } else {
        std::cerr << "sec_ is not a tensor list\n";
    }
    sec.resize(values.size());
    for (int ii=0; ii<values.size(); ii++) {
        sec[ii] = static_cast<int>(values[ii]);
    }
    */
    // rcut = 6.0;
    // sec = std::vector<int> {46, 138};
}

class DeepPotModelDevi {
 public:
  /**
   * @brief DP model deviation constructor without initialization.
   **/
  DeepPotModelDevi();
  ~DeepPotModelDevi();
  /**
   * @brief Initialize the DP.
   * @param[in] models The name of the frozen model files.
   **/
  template <typename VALUETYPE>
  void init(const std::vector<std::string>& models);

  int numb_types() { return sec.size(); }

  /**
   * @brief Evaluate the energy, force and virial by using this DP.
   * @param[out] all_ener The system energy.
   * @param[out] all_force The force on each atom.
   * @param[out] all_virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(std::vector<ENERGYVTYPE>& all_ener,
               std::vector<std::vector<VALUETYPE>>& all_force,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box);
 private:
  unsigned numb_models;
  std::vector<torch::jit::script::Module> modules;
  double rcut;
  std::vector<int> sec;
};

template <typename VALUETYPE>
void DeepPotModelDevi::init(const std::vector<std::string>& models) {
    numb_models = models.size();
    modules.resize(numb_models);
    // cublasCreate(&handle);

    try {
        for (int ii=0; ii<numb_models; ii++) {
            modules[ii] = torch::jit::load(models[ii]);
        }
        
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
    }


    auto rcut_ = modules[0].run_method("get_rcut").toDouble();
    rcut = static_cast<VALUETYPE>(rcut_);

    auto sec_ = modules[0].run_method("get_sec");
    TensortoVec<int64_t, int>(sec_, sec);
}
}  // namespace deepmd