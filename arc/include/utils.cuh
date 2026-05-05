#pragma once

#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cutlass/detail/helper_macros.hpp"

namespace arc {
template <class a_type, class b_type, class c_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major>
struct MmaInstr {
  using Op = cute::SM100_MMA_F16BF16_SS<a_type, b_type, c_type, M, N, a_major,
                                        b_major>;

  CUTE_HOST_DEVICE static constexpr uint64_t idesc() {
    return cute::UMMA::make_runtime_instr_desc<a_type, b_type, c_type, M, N,
                                               a_major, b_major>();
  }

  CUTLASS_DEVICE static void fma(uint64_t desc_a, uint64_t desc_b,
                                 uint32_t tmem_c, uint32_t scale_c,
                                 uint64_t runtime_instr_desc) {
    Op::fma(desc_a, desc_b, tmem_c, scale_c, runtime_instr_desc);
  }
};
} // namespace arc
