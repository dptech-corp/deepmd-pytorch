#ifdef PAIR_CLASS
// clang-format off
PairStyle(deepmd,PairDeepMD);
// clang-format on
#else

#ifndef LMP_PAIR_DEEPMD_H
#define LMP_PAIR_DEEPMD_H

#include "pair.h"
#include "DeepPot.h"


namespace LAMMPS_NS {

class PairDeepMD : public Pair {
 public:
  PairDeepMD(class LAMMPS *);
  ~PairDeepMD() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
//   void init_style() override;

 protected:
  virtual void allocate();
  double **scale;

 private:
  deepmd::DeepPot deep_pot;
  deepmd::DeepPotModelDevi deep_pot_model_devi;
  unsigned numb_models;
  double cutoff;
  int numb_types;
  int out_freq;
  std::string out_file;
  int dim_fparam;
  int dim_aparam;
  int out_each;
  int out_rel;
  int out_rel_v;
  double eps;
};

}    // namespace LAMMPS_NS

#endif
#endif
