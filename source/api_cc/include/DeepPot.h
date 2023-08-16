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
  double rcut;
  std::vector<int> sec;
};
}  // namespace deepmd