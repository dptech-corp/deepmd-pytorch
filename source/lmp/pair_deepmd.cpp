#include "pair_deepmd.h"

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
    models.push_back(arg[ii]);
  }
  numb_models = models.size();
  if (numb_models == 1) {
    // try {
      deep_pot.init(arg[0]);
    // } catch (deepmd_compat::deepmd_exception &e) {
      // error->one(FLERR, e.what());
    // }
  }
}

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
    dtype[ii] = type[ii];
  }

  // get box
  dbox[0] = domain->h[0];  // xx
  dbox[4] = domain->h[1];  // yy
  dbox[8] = domain->h[2];  // zz
  dbox[7] = domain->h[3];  // zy
  dbox[6] = domain->h[4];  // zx
  dbox[3] = domain->h[5];  // yx

  deep_pot.compute(dener, dforce, dvirial, dcoord, dtype, dbox);

  for (int ii=0; ii<nlocal; ii++) {
    for (int jj=0; jj<3; jj++) {
      f[ii][jj] = dforce[3*ii+jj];
    }
  }

  // accumulate energy and virial
  if (eflag) {
    eng_vdwl += scale[1][1] * dener;
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