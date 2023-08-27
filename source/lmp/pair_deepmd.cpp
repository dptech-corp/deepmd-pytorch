#include "pair_deepmd.h"

#include <string.h>

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "output.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace std;

static bool is_key(const string &input) {
  vector<string> keys;
  keys.push_back("out_freq");
  keys.push_back("out_file");
  keys.push_back("fparam");
  keys.push_back("aparam");
  keys.push_back("fparam_from_compute");
  keys.push_back("ttm");
  keys.push_back("atomic");
  keys.push_back("relative");
  keys.push_back("relative_v");
  keys.push_back("virtual_len");
  keys.push_back("spin_norm");

  for (int ii = 0; ii < keys.size(); ++ii) {
    if (input == keys[ii]) {
      return true;
    }
  }
  return false;
}

PairDeepMD::PairDeepMD(LAMMPS *lmp)
    : Pair(lmp)

{
  if (strcmp(update->unit_style, "metal") != 0) {
    error->all(
        FLERR,
        "Pair deepmd requires metal unit, please set it by \"units metal\"");
  }
  numb_models = 0;

}

/* ---------------------------------------------------------------------- */

PairDeepMD::~PairDeepMD() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::settings(int narg, char **arg) {
  if (narg <= 0) {
    error->all(FLERR, "Illegal pair_style command");
  }

  vector<string> models;
  int iarg = 0;
  while (iarg < narg) {
    if (is_key(arg[iarg])) {
      break;
    }
    iarg++;
  }
  for (int ii = 0; ii < iarg; ++ii) {
    models.push_back(std::string(arg[ii]));
  }
  numb_models = models.size();
  if (numb_models == 1) {
    // try {
      deep_pot.init<double>(std::string(arg[0]));
      numb_types = deep_pot.numb_types();
    // } catch (deepmd_compat::deepmd_exception &e) {
      // error->one(FLERR, e.what());
    // }
  }
  else {
    try {
      deep_pot.init(std::string(arg[0]));
      deep_pot_model_devi.init(models, get_node_rank(),
                              get_file_content(models));
    } catch (deepmd_compat::deepmd_exception &e) {
      error->one(FLERR, e.what());
    }
    cutoff = deep_pot_model_devi.cutoff();
    numb_types = deep_pot_model_devi.numb_types();
    numb_types_spin = deep_pot_model_devi.numb_types_spin();
    dim_fparam = deep_pot_model_devi.dim_fparam();
    dim_aparam = deep_pot_model_devi.dim_aparam();
    assert(cutoff == deep_pot.cutoff());
    assert(numb_types == deep_pot.numb_types());
    assert(numb_types_spin == deep_pot.numb_types_spin());
    assert(dim_fparam == deep_pot.dim_fparam());
    assert(dim_aparam == deep_pot.dim_aparam());
  }

  out_freq = 100;
  out_file = "model_devi.out";
  out_each = 0;
  out_rel = 0;
  eps = 0.;
  fparam.clear();
  aparam.clear();
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::compute(int eflag, int vflag) {
  if (numb_models == 0) {
    return;
  }
  if (eflag || vflag) {
    ev_setup(eflag, vflag);
  }
  if (vflag_atom) {
    error->all(FLERR,
               "6-element atomic virial is not supported. Use compute "
               "centroid/stress/atom command for 9-element atomic virial.");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double dener(0);
  vector<double> dforce(nlocal * 3);
  vector<double> dvirial(9, 0);
  vector<double> dcoord(nlocal * 3, 0.);
  vector<int> dtype(nlocal);
  vector<double> dbox(9, 0);

  for (int ii=0; ii<nlocal; ii++) {
    for (int jj=0; jj<3; jj++) {
      dcoord[3*ii+jj] = x[ii][jj];
    }
    dtype[ii] = type[ii]-1;
  }

  // get box
  dbox[0] = domain->h[0];  // xx
  dbox[4] = domain->h[1];  // yy
  dbox[8] = domain->h[2];  // zz
  dbox[7] = domain->h[3];  // zy
  dbox[6] = domain->h[4];  // zx
  dbox[3] = domain->h[5];  // yx

  deep_pot.compute<double, double>(dener, dforce, dvirial, dcoord, dtype, dbox);

  for (int ii=0; ii<nlocal; ii++) {
    for (int jj=0; jj<3; jj++) {
      f[ii][jj] = dforce[3*ii+jj];
    }
  }

  // accumulate energy and virial
  if (eflag) {
    eng_vdwl += dener;
  }
  if (vflag) {
    virial[0] += 1.0 * dvirial[0];
    virial[1] += 1.0 * dvirial[4];
    virial[2] += 1.0 * dvirial[8];
    virial[3] += 1.0 * dvirial[3];
    virial[4] += 1.0 * dvirial[6];
    virial[5] += 1.0 * dvirial[7];
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDeepMD::coeff(int narg, char **arg) {
  // if (narg < 4 || narg > 5) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      // epsilon[i][j] = epsilon_one;
      // sigma[i][j] = sigma_one;
      // cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}


/* ---------------------------------------------------------------------- */

void PairDeepMD::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(scale, n + 1, n + 1, "pair:scale");

  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      setflag[i][j] = 0;
      scale[i][j] = 0;
    }
  }
  for (int i = 1; i <= numb_types; ++i) {
    if (i > n) {
      continue;
    }
    for (int j = i; j <= numb_types; ++j) {
      if (j > n) {
        continue;
      }
      setflag[i][j] = 1;
      scale[i][j] = 1;
    }
  }
}
