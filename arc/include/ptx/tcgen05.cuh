#pragma once
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/util.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>

namespace arc::tcgen05 {
namespace detail {
template <uint32_t Dp, uint32_t Bits> struct UTCCPOpSelector {
  static_assert(Dp != Dp, "Unsupported tcgen05.cp UTCCPOpSelector<dp, bits>");
  static constexpr uint32_t kCols = 0;
  static constexpr uint32_t kBytes = 0;
};

template <> struct UTCCPOpSelector<128, 128> {
  using Op = cute::SM100::TMEM::UTCCP::SM100_UTCCP_128dp128bit_1cta;
  static constexpr uint32_t kCols = 128 / 32;
  static constexpr uint32_t kBytes = 128 * 128 / 8;
};

template <> struct UTCCPOpSelector<128, 256> {
  using Op = cute::SM100::TMEM::UTCCP::SM100_UTCCP_128dp256bit_1cta;
  static constexpr uint32_t kCols = 256 / 32;
  static constexpr uint32_t kBytes = 128 * 256 / 8;
};

template <> struct UTCCPOpSelector<4, 256> {
  using Op = cute::SM100::TMEM::UTCCP::SM100_UTCCP_4dp256bit_1cta;
  static constexpr uint32_t kCols = 1;
  static constexpr uint32_t kBytes = 4 * 256 / 8;
};

template <> struct UTCCPOpSelector<32, 128> {
  using Op = cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta;
  static constexpr uint32_t kCols = 128 / 32;
  static constexpr uint32_t kBytes = 4 * 32 * 128 / 8;
};

template <uint32_t Dp, uint32_t Bits, uint32_t RepeatTimes>
struct UTCLdOpSelector {
  static_assert(
      Dp != Dp,
      "Unsupported tcgen05.ld UTCLdOpSelector<dp, bits, repeat_times>");
  static constexpr uint32_t kCols = 0;
  static constexpr uint32_t kBytes = 0;
};

template <> struct UTCLdOpSelector<32, 32, 1> {
  using Op = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b1x;
  static constexpr uint32_t kCols = 1;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 1 / 8;
};

template <> struct UTCLdOpSelector<32, 32, 2> {
  using Op = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b2x;
  static constexpr uint32_t kCols = 2;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 2 / 8;
};

template <> struct UTCLdOpSelector<32, 32, 4> {
  using Op = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b4x;
  static constexpr uint32_t kCols = 4;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 4 / 8;
};

template <> struct UTCLdOpSelector<32, 32, 8> {
  using Op = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b8x;
  static constexpr uint32_t kCols = 8;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 8 / 8;
};

template <> struct UTCLdOpSelector<32, 32, 16> {
  using Op = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x;
  static constexpr uint32_t kCols = 16;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 16 / 8;
};

template <> struct UTCLdOpSelector<32, 32, 32> {
  using Op = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x;
  static constexpr uint32_t kCols = 32;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 32 / 8;
};

template <uint32_t Dp, uint32_t Bits, uint32_t RepeatTimes>
struct UTCStOpSelector {
  static_assert(
      Dp != Dp,
      "Unsupported tcgen05.st UTCStOpSelector<dp, bits, repeat_times>");
  static constexpr uint32_t kCols = 0;
  static constexpr uint32_t kBytes = 0;
};

template <> struct UTCStOpSelector<32, 32, 1> {
  using Op = cute::SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b1x;
  static constexpr uint32_t kCols = 1;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 1 / 8;
};

template <> struct UTCStOpSelector<32, 32, 2> {
  using Op = cute::SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b2x;
  static constexpr uint32_t kCols = 2;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 2 / 8;
};

template <> struct UTCStOpSelector<32, 32, 4> {
  using Op = cute::SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b4x;
  static constexpr uint32_t kCols = 4;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 4 / 8;
};

template <> struct UTCStOpSelector<32, 32, 8> {
  using Op = cute::SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b8x;
  static constexpr uint32_t kCols = 8;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 8 / 8;
};

template <> struct UTCStOpSelector<32, 32, 16> {
  using Op = cute::SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b16x;
  static constexpr uint32_t kCols = 16;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 16 / 8;
};

template <> struct UTCStOpSelector<32, 32, 32> {
  using Op = cute::SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b32x;
  static constexpr uint32_t kCols = 32;
  static constexpr uint32_t kBytes = 4 * 32 * 32 * 32 / 8;
};

template <typename LdOp, uint32_t RepeatTimes, std::size_t... I>
CUTLASS_DEVICE void utcld_impl(uint32_t src_addr, uint32_t (&r)[RepeatTimes],
                               std::index_sequence<I...>) {
  LdOp::copy(src_addr, r[I]...);
}

template <typename StOp, uint32_t RepeatTimes, std::size_t... I>
CUTLASS_DEVICE void utcst_impl(uint32_t dst_addr,
                               uint32_t const (&r)[RepeatTimes],
                               std::index_sequence<I...>) {
  StOp::copy(r[I]..., dst_addr);
}
} // namespace detail

template <uint32_t Dp, uint32_t Bits>
CUTLASS_DEVICE void utccp(uint32_t dst_addr, uint64_t src_desc) {
  using CopyOp = typename detail::UTCCPOpSelector<Dp, Bits>::Op;
  CopyOp::copy(src_desc, dst_addr);
}

template <uint32_t Dp, uint32_t Bits, uint32_t RepeatTimes>
CUTLASS_DEVICE void utcld(uint32_t src_addr, uint32_t (&r)[RepeatTimes]) {
  using LdOp = typename detail::UTCLdOpSelector<Dp, Bits, RepeatTimes>::Op;
  detail::utcld_impl<LdOp>(src_addr, r,
                           std::make_index_sequence<RepeatTimes>{});
}

template <uint32_t Dp, uint32_t Bits, uint32_t RepeatTimes>
CUTLASS_DEVICE void utcst(uint32_t dst_addr, uint32_t const (&r)[RepeatTimes]) {
  using StOp = typename detail::UTCStOpSelector<Dp, Bits, RepeatTimes>::Op;
  detail::utcst_impl<StOp>(dst_addr, r,
                           std::make_index_sequence<RepeatTimes>{});
}

namespace detail {
CUTLASS_DEVICE cute::UMMA::SmemDescriptor
make_smem_desc(cute::UMMA::LayoutType layout_type, void *smem_ptr, uint32_t sbo,
               uint32_t lbo) {
  cute::UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = static_cast<uint8_t>(layout_type);
  desc.start_address_ =
      static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(smem_ptr) >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = sbo >> 4;
  desc.leading_byte_offset_ = lbo >> 4;
  return desc;
}
} // namespace detail

// (BLOCK_MN, BK) -> (BK / _BK, BLOCK_MN, _BK)
template <uint32_t BLOCK_MN, uint32_t BK, typename dtype_t,
          uint32_t kSwizzleBytes = 128, uint32_t kAtomBytes = 16>
CUTLASS_DEVICE cute::UMMA::SmemDescriptor
make_smem_desc(dtype_t *base, uint32_t mn_idx, uint32_t k_idx) {

  constexpr uint32_t _BK = kSwizzleBytes / sizeof(dtype_t);
  static_assert(BK % _BK == 0);

  cute::UMMA::LayoutType layout =
      kAtomBytes == 32       ? cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B
      : kSwizzleBytes == 128 ? cute::UMMA::LayoutType::SWIZZLE_128B
      : kSwizzleBytes == 64  ? cute::UMMA::LayoutType::SWIZZLE_64B
      : kSwizzleBytes == 32  ? cute::UMMA::LayoutType::SWIZZLE_32B
                             : cute::UMMA::LayoutType::SWIZZLE_NONE;

  uint32_t k_slab = k_idx / _BK;
  uint32_t k_in_slab = k_idx % _BK;
  dtype_t *ptr = base + k_slab * BLOCK_MN * _BK + mn_idx * _BK + k_in_slab;
  return detail::make_smem_desc(layout, ptr, 8 * kSwizzleBytes, kAtomBytes);
}

} // namespace arc::tcgen05
