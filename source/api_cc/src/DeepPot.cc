#include "DeepPot.h"

namespace deepmd{
DeepPot::DeepPot() {}

DeepPot::~DeepPot() {  }//cublasDestroy(handle);}



template void DeepPot::init<double>(const std::string& model);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPot::compute(ENERGYVTYPE& ener,
            std::vector<VALUETYPE>& force,
            std::vector<VALUETYPE>& virial,
            const std::vector<VALUETYPE>& coord,
            const std::vector<int>& atype,
            const std::vector<VALUETYPE>& box)
{
    auto device = torch::kCUDA;
    module.to(device);

    std::vector<std::vector<int>> nlist, nlist_loc, nlist_type;
    std::vector<int> merged_mapping;
    std::vector<VALUETYPE> merged_coord_shift;
    std::vector<VALUETYPE> coord_wrapped;

    int ntype = sec.size();
    std::vector<int> natoms(2+ntype);
    natoms[0] = atype.size();
    natoms[1] = atype.size();
    for (int ii=0; ii<ntype; ii++) {
      natoms[ii+2] = std::count(atype.begin(), atype.end(), ii);
    }

    make_env_mat(nlist, nlist_loc, nlist_type, merged_coord_shift, merged_mapping, coord_wrapped, coord, atype, box, rcut, sec);
    int nall = merged_mapping.size();
    // auto device = torch::kCPU;
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto int_options = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<torch::jit::IValue> inputs;

    at::Tensor coord_wrapped_Tensor = torch::from_blob(coord_wrapped.data(), {1, natoms[0], 3}, options).to(device);
    // for (int ii=0; ii<natoms[0]; ii++) { printf("%.4e %.4e %.4e\n", coord_wrapped[3*ii+0], coord_wrapped[3*ii+1], coord_wrapped[3*ii+2]); } printf("\n");
    inputs.push_back(coord_wrapped_Tensor);

    std::vector<int64_t> atype_64(atype.begin(), atype.end());
    at::Tensor atype_Tensor = torch::from_blob(atype_64.data(), {1, natoms[0]}, int_options).to(device);
    inputs.push_back(atype_Tensor);

    std::vector<int64_t> natoms_64(natoms.begin(), natoms.end());
    at::Tensor natoms_Tensor = torch::from_blob(natoms_64.data(), {1, 2+ntype}, int_options).to(device);
    inputs.push_back(natoms_Tensor);

    std::vector<int64_t> merged_mapping_64(merged_mapping.begin(), merged_mapping.end());
    at::Tensor merged_mapping_Tensor = torch::from_blob(merged_mapping_64.data(), {1, nall}, int_options).to(device);
    // for (int ii=0; ii<merged_mapping.size(); ii++) { printf("%d ", merged_mapping[ii]); } printf("\n");
    inputs.push_back(merged_mapping_Tensor);

    at::Tensor merged_coord_shift_Tensor = torch::from_blob(merged_coord_shift.data(), {1, nall, 3}, options).to(device);
    inputs.push_back(merged_coord_shift_Tensor);

   std::vector<int64_t> nlist_64(natoms[0]*sec[ntype-1]);
    for (int ii=0; ii<natoms[0]; ii++) {
        for (int jj=0; jj<sec[ntype-1]; jj++) {
            nlist_64[ii*sec[ntype-1]+jj] = nlist[ii][jj];
        }
    }
    at::Tensor nlist_Tensor = torch::from_blob(nlist_64.data(), {1, natoms[0], sec[ntype-1]}, int_options).to(device);
    inputs.push_back(nlist_Tensor);

    std::vector<int64_t> nlist_loc_64(natoms[0]*sec[ntype-1]);
    for (int ii=0; ii<natoms[0]; ii++) {
        for (int jj=0; jj<sec[ntype-1]; jj++) {
            nlist_loc_64[ii*sec[ntype-1]+jj] = nlist_loc[ii][jj];
        }
    }
    at::Tensor nlist_loc_Tensor = torch::from_blob(nlist_loc_64.data(), {1, natoms[0], sec[ntype-1]}, int_options).to(device);
    inputs.push_back(nlist_loc_Tensor);

    std::vector<int64_t> nlist_type_64(natoms[0]*sec[ntype-1]);
    for (int ii=0; ii<natoms[0]; ii++) {
        for (int jj=0; jj<sec[ntype-1]; jj++) {
            nlist_type_64[ii*sec[ntype-1]+jj] = nlist_type[ii][jj];
        }
    }
    at::Tensor nlist_type_Tensor = torch::from_blob(nlist_type_64.data(), {1, natoms[0], sec[ntype-1]}, int_options).to(device);
    inputs.push_back(nlist_type_Tensor);

    at::Tensor box_Tensor = torch::from_blob(const_cast<VALUETYPE*>(box.data()), {1, 9}, options).to(device);
    inputs.push_back(box_Tensor);
    c10::Dict<c10::IValue, c10::IValue> outputs = module.forward(inputs).toGenericDict();

    c10::IValue energy_ = outputs.at("energy");
    c10::IValue force_ = outputs.at("force");
    c10::IValue virial_ = outputs.at("virial");
    // std::cout << energy_ << std::endl;
    // std::cout << force_ << std::endl;
    ener = energy_.toTensor().item<double>();

    torch::Tensor flat_force_ = force_.toTensor().view({-1});
    torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
    force.assign(cpu_force_.data_ptr<double>(), cpu_force_.data_ptr<double>() + cpu_force_.numel());

    torch::Tensor flat_virial_ = virial_.toTensor().view({-1});
    torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
    virial.assign(cpu_virial_.data_ptr<double>(), cpu_virial_.data_ptr<double>() + cpu_virial_.numel());

}
template void DeepPot::compute<double, double>(double& ener,
            std::vector<double>& force,
            std::vector<double>& virial,
            const std::vector<double>& coord,
            const std::vector<int>& atype,
            const std::vector<double>& box);

template void DeepPotModelDevi::init<double>(const std::vector<std::string>& models);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotModelDevi::compute(std::vector<ENERGYVTYPE>& all_ener,
            std::vector<std::vector<VALUETYPE>>& all_force,
            std::vector<std::vector<VALUETYPE>>& all_virial,
            const std::vector<VALUETYPE>& coord,
            const std::vector<int>& atype,
            const std::vector<VALUETYPE>& box)
{
    auto device = torch::kCUDA;
    for (int ii=0; ii<numb_models; ii++) {
        modules[i].to(device);
    }

    std::vector<std::vector<int>> nlist, nlist_loc, nlist_type;
    std::vector<int> merged_mapping;
    std::vector<VALUETYPE> merged_coord_shift;
    std::vector<VALUETYPE> coord_wrapped;

    int ntype = sec.size();
    std::vector<int> natoms(2+ntype);
    natoms[0] = atype.size();
    natoms[1] = atype.size();
    for (int ii=0; ii<ntype; ii++) {
      natoms[ii+2] = std::count(atype.begin(), atype.end(), ii);
    }

    make_env_mat(nlist, nlist_loc, nlist_type, merged_coord_shift, merged_mapping, coord_wrapped, coord, atype, box, rcut, sec);
    int nall = merged_mapping.size();
    // auto device = torch::kCPU;
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto int_options = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<torch::jit::IValue> inputs;

    at::Tensor coord_wrapped_Tensor = torch::from_blob(coord_wrapped.data(), {1, natoms[0], 3}, options).to(device);
    // for (int ii=0; ii<natoms[0]; ii++) { printf("%.4e %.4e %.4e\n", coord_wrapped[3*ii+0], coord_wrapped[3*ii+1], coord_wrapped[3*ii+2]); } printf("\n");
    inputs.push_back(coord_wrapped_Tensor);

    std::vector<int64_t> atype_64(atype.begin(), atype.end());
    at::Tensor atype_Tensor = torch::from_blob(atype_64.data(), {1, natoms[0]}, int_options).to(device);
    inputs.push_back(atype_Tensor);

    std::vector<int64_t> natoms_64(natoms.begin(), natoms.end());
    at::Tensor natoms_Tensor = torch::from_blob(natoms_64.data(), {1, 2+ntype}, int_options).to(device);
    inputs.push_back(natoms_Tensor);

    std::vector<int64_t> merged_mapping_64(merged_mapping.begin(), merged_mapping.end());
    at::Tensor merged_mapping_Tensor = torch::from_blob(merged_mapping_64.data(), {1, nall}, int_options).to(device);
    // for (int ii=0; ii<merged_mapping.size(); ii++) { printf("%d ", merged_mapping[ii]); } printf("\n");
    inputs.push_back(merged_mapping_Tensor);

    at::Tensor merged_coord_shift_Tensor = torch::from_blob(merged_coord_shift.data(), {1, nall, 3}, options).to(device);
    inputs.push_back(merged_coord_shift_Tensor);

   std::vector<int64_t> nlist_64(natoms[0]*sec[ntype-1]);
    for (int ii=0; ii<natoms[0]; ii++) {
        for (int jj=0; jj<sec[ntype-1]; jj++) {
            nlist_64[ii*sec[ntype-1]+jj] = nlist[ii][jj];
        }
    }
    at::Tensor nlist_Tensor = torch::from_blob(nlist_64.data(), {1, natoms[0], sec[ntype-1]}, int_options).to(device);
    inputs.push_back(nlist_Tensor);

    std::vector<int64_t> nlist_loc_64(natoms[0]*sec[ntype-1]);
    for (int ii=0; ii<natoms[0]; ii++) {
        for (int jj=0; jj<sec[ntype-1]; jj++) {
            nlist_loc_64[ii*sec[ntype-1]+jj] = nlist_loc[ii][jj];
        }
    }
    at::Tensor nlist_loc_Tensor = torch::from_blob(nlist_loc_64.data(), {1, natoms[0], sec[ntype-1]}, int_options).to(device);
    inputs.push_back(nlist_loc_Tensor);

    std::vector<int64_t> nlist_type_64(natoms[0]*sec[ntype-1]);
    for (int ii=0; ii<natoms[0]; ii++) {
        for (int jj=0; jj<sec[ntype-1]; jj++) {
            nlist_type_64[ii*sec[ntype-1]+jj] = nlist_type[ii][jj];
        }
    }
    at::Tensor nlist_type_Tensor = torch::from_blob(nlist_type_64.data(), {1, natoms[0], sec[ntype-1]}, int_options).to(device);
    inputs.push_back(nlist_type_Tensor);

    at::Tensor box_Tensor = torch::from_blob(const_cast<VALUETYPE*>(box.data()), {1, 9}, options).to(device);
    inputs.push_back(box_Tensor);
    c10::Dict<c10::IValue, c10::IValue> outputs = module.forward(inputs).toGenericDict();

    c10::IValue energy_ = outputs.at("energy");
    c10::IValue force_ = outputs.at("force");
    c10::IValue virial_ = outputs.at("virial");
    // std::cout << energy_ << std::endl;
    // std::cout << force_ << std::endl;
    ener = energy_.toTensor().item<double>();

    torch::Tensor flat_force_ = force_.toTensor().view({-1});
    torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
    force.assign(cpu_force_.data_ptr<double>(), cpu_force_.data_ptr<double>() + cpu_force_.numel());

    torch::Tensor flat_virial_ = virial_.toTensor().view({-1});
    torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
    virial.assign(cpu_virial_.data_ptr<double>(), cpu_virial_.data_ptr<double>() + cpu_virial_.numel());

}
template void DeepPotModelDevi::compute<double, double>(std::vector<double>& all_ener,
            std::vector<std::vector<double>>& all_force,
            std::vector<std::vector<double>>& all_virial,
            const std::vector<double>& coord,
            const std::vector<int>& atype,
            const std::vector<double>& box);

}