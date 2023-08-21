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
//   void coeff(int, char **) override;
//   void init_style() override;

 protected:
  deepmd::DeepPot deep_pot;
};

}    // namespace LAMMPS_NS

#endif
#endif