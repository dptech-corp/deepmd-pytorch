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

  /**
   * @brief Evaluate the energy, force and virial by using this DP.
   * @param[out] all_energy The system energy.
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
  /**
   * @brief Compute the average energy.
   * @param[out] dener The average energy.
   * @param[in] all_energy The energies of all models.
   **/
  template <typename VALUETYPE>
  void compute_avg(VALUETYPE& dener, const std::vector<VALUETYPE>& all_energy);
  /**
   * @brief Compute the average of vectors.
   * @param[out] avg The average of vectors.
   * @param[in] xx The vectors of all models.
   **/
  template <typename VALUETYPE>
  void compute_avg(std::vector<VALUETYPE>& avg,
                   const std::vector<std::vector<VALUETYPE> >& xx);
 /**
   * @brief Compute the standard deviation of vectors.
   * @param[out] std The standard deviation of vectors.
   * @param[in] avg The average of vectors.
   * @param[in] xx The vectors of all models.
   * @param[in] stride The stride to compute the deviation.
   **/
  template <typename VALUETYPE>
  void compute_std(std::vector<VALUETYPE>& std,
                   const std::vector<VALUETYPE>& avg,
                   const std::vector<std::vector<VALUETYPE> >& xx,
                   const int& stride);
  /**
   * @brief Compute the relative standard deviation of vectors.
   * @param[out] std The standard deviation of vectors.
   * @param[in] avg The average of vectors.
   * @param[in] eps The level parameter for computing the deviation.
   * @param[in] stride The stride to compute the deviation.
   **/
  template <typename VALUETYPE>
  void compute_relative_std(std::vector<VALUETYPE>& std,
                            const std::vector<VALUETYPE>& avg,
                            const VALUETYPE eps,
                            const int& stride);
  /**
   * @brief Compute the standard deviation of forces.
   * @param[out] std The standard deviation of forces.
   * @param[in] avg The average of forces.
   * @param[in] xx The vectors of all forces.
   **/
  template <typename VALUETYPE>
  void compute_std_f(std::vector<VALUETYPE>& std,
                     const std::vector<VALUETYPE>& avg,
                     const std::vector<std::vector<VALUETYPE> >& xx);
  /**
   * @brief Compute the relative standard deviation of forces.
   * @param[out] std The relative standard deviation of forces.
   * @param[in] avg The relative average of forces.
   * @param[in] eps The level parameter for computing the deviation.
   **/
  template <typename VALUETYPE>
  void compute_relative_std_f(std::vector<VALUETYPE>& std,
                              const std::vector<VALUETYPE>& avg,
                              const VALUETYPE eps);

 private:
  unsigned numb_models;
  std::vector<torch::jit::script::Module> modules;
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
}
}  // namespace deepmd