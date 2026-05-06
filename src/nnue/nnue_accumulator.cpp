/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "nnue_accumulator.h"

#include <cassert>
#include <cstdint>
#include <new>
#include <type_traits>

#include "../bitboard.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "network.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_feature_transformer.h"  // IWYU pragma: keep
#include "simd.h"

namespace Stockfish::Eval::NNUE {

using namespace SIMD;

namespace {

template<typename FeatureSet> struct AccumulatorUpdateContext;

inline Bitboard get_changed_pieces(const std::array<Piece, SQUARE_NB>& oldPieces,
                            const std::array<Piece, SQUARE_NB>& newPieces);

}

template<typename T>
const AccumulatorState<T>& AccumulatorStack::latest() const noexcept {
    return accumulators<T>()[size - 1];
}

// Explicit template instantiations
template const AccumulatorState<PSQFeatureSet>&    AccumulatorStack::latest() const noexcept;
template const AccumulatorState<ThreatFeatureSet>& AccumulatorStack::latest() const noexcept;

AccumulatorStack::AccumulatorStack(const Network& network)
    : ft(network.featureTransformer) {}

template<typename T>
AccumulatorState<T>& AccumulatorStack::mut_latest() noexcept {
    return mut_accumulators<T>()[size - 1];
}

template<typename T>
const std::array<AccumulatorState<T>, AccumulatorStack::MaxSize>&
AccumulatorStack::accumulators() const noexcept {
    static_assert(std::is_same_v<T, PSQFeatureSet> || std::is_same_v<T, ThreatFeatureSet>,
                  "Invalid Feature Set Type");

    if constexpr (std::is_same_v<T, PSQFeatureSet>)
        return psq_accumulators;

    if constexpr (std::is_same_v<T, ThreatFeatureSet>)
        return threat_accumulators;
}

template<typename T>
std::array<AccumulatorState<T>, AccumulatorStack::MaxSize>&
AccumulatorStack::mut_accumulators() noexcept {
    static_assert(std::is_same_v<T, PSQFeatureSet> || std::is_same_v<T, ThreatFeatureSet>,
                  "Invalid Feature Set Type");

    if constexpr (std::is_same_v<T, PSQFeatureSet>)
        return psq_accumulators;

    if constexpr (std::is_same_v<T, ThreatFeatureSet>)
        return threat_accumulators;
}

void AccumulatorStack::reset() noexcept {
    psq_accumulators[0].reset({});
    threat_accumulators[0].reset({});
    size = 1;
}

std::pair<DirtyPiece&, DirtyThreats&> AccumulatorStack::push() noexcept {
    assert(size < MaxSize);
    auto& dp  = psq_accumulators[size].reset();
    auto& dts = threat_accumulators[size].reset();
    new (&dts) DirtyThreats;
    size++;
    return {dp, dts};
}

void AccumulatorStack::pop() noexcept {
    assert(size > 1);
    size--;
}

template<typename FeatureSet>
void AccumulatorStack::evaluate_side(Color                     perspective,
                                     const Position&           pos,
                                     AccumulatorCaches&        cache) noexcept {

    const auto last_usable_accum = find_last_usable_accumulator<FeatureSet>(perspective);

    if (accumulators<FeatureSet>()[last_usable_accum].computed[perspective])
        forward_update_incremental<FeatureSet>(perspective, pos,
                                               last_usable_accum);

    else
    {
        if constexpr (std::is_same_v<FeatureSet, PSQFeatureSet>)
            update_accumulator_refresh_cache(perspective, pos,
                                             mut_latest<PSQFeatureSet>(), cache);
        else
            update_threats_accumulator_full(perspective, pos,
                                            mut_latest<ThreatFeatureSet>());

        backward_update_incremental<FeatureSet>(perspective, pos,
                                                last_usable_accum);
    }
}

// Find the earliest usable accumulator, this can either be a computed accumulator or the accumulator
// state just before a change that requires full refresh.
template<typename FeatureSet>
std::size_t AccumulatorStack::find_last_usable_accumulator(Color perspective) const noexcept {

    for (std::size_t curr_idx = size - 1; curr_idx > 0; curr_idx--)
    {
        if (accumulators<FeatureSet>()[curr_idx].computed[perspective])
            return curr_idx;

        if (FeatureSet::requires_refresh(accumulators<FeatureSet>()[curr_idx].diff, perspective))
            return curr_idx;
    }

    return 0;
}

template<typename FeatureSet>
void AccumulatorStack::forward_update_incremental(Color                     perspective,
                                                  const Position&           pos,
                                                  const std::size_t         begin) noexcept {

    assert(begin < accumulators<FeatureSet>().size());
    assert(accumulators<FeatureSet>()[begin].computed[perspective]);

    const Square ksq = pos.square<KING>(perspective);

    for (std::size_t next = begin + 1; next < size; next++)
    {
        update_accumulator_incremental<true>(perspective, ksq,
                                             mut_accumulators<FeatureSet>()[next],
                                             accumulators<FeatureSet>()[next - 1]);
    }

    assert(latest<FeatureSet>().computed[perspective]);
}

template<typename FeatureSet>
void AccumulatorStack::backward_update_incremental(Color perspective,
                                                   const Position&           pos,
                                                   const std::size_t         end) noexcept {

    assert(end < accumulators<FeatureSet>().size());
    assert(end < size);
    assert(latest<FeatureSet>().computed[perspective]);

    const Square ksq = pos.square<KING>(perspective);

    for (std::int64_t next = std::int64_t(size) - 2; next >= std::int64_t(end); next--)
        update_accumulator_incremental<false>(perspective, ksq,
                                              mut_accumulators<FeatureSet>()[next],
                                              accumulators<FeatureSet>()[next + 1]);

    assert(accumulators<FeatureSet>()[end].computed[perspective]);
}


template<bool Forward, typename FeatureSet>
void AccumulatorStack::update_accumulator_incremental(Color                               perspective,
                                                      const Square                        ksq,
                                                      AccumulatorState<FeatureSet>&       target_state,
                                                      const AccumulatorState<FeatureSet>& computed) {

    assert(computed.computed[perspective]);
    assert(!target_state.computed[perspective]);

    // The size must be enough to contain the largest possible update.
    // That might depend on the feature set and generally relies on the
    // feature set's update cost calculation to be correct and never allow
    // updates with more added/removed features than MaxActiveDimensions.
    // In this case, the maximum size of both feature addition and removal
    // is 2, since we are incrementally updating one move at a time.
    typename FeatureSet::IndexList removed, added;
    if constexpr (std::is_same_v<FeatureSet, ThreatFeatureSet>)
    {
        const auto* pfBase   = &ft.threatWeights[0];
        IndexType   pfStride = L1;
        if constexpr (Forward)
            FeatureSet::append_changed_indices(perspective, ksq, target_state.diff, removed, added,
                                               nullptr, false, pfBase, pfStride);
        else
            FeatureSet::append_changed_indices(perspective, ksq, computed.diff, added, removed,
                                               nullptr, false, pfBase, pfStride);
    }
    else
    {
        if constexpr (Forward)
            FeatureSet::append_changed_indices(perspective, ksq, target_state.diff, removed, added);
        else
            FeatureSet::append_changed_indices(perspective, ksq, computed.diff, added, removed);
    }

    AccumulatorUpdateContext<FeatureSet> updateContext{perspective, ft, computed, target_state};

    if constexpr (std::is_same_v<FeatureSet, ThreatFeatureSet>)
        updateContext.apply(added, removed);
    else
    {
        [[maybe_unused]] const int addedSize   = added.ssize();
        [[maybe_unused]] const int removedSize = removed.ssize();

        assert(addedSize == 1 || addedSize == 2);
        assert(removedSize == 1 || removedSize == 2);
        assert((Forward && addedSize <= removedSize) || (!Forward && addedSize >= removedSize));

        // Workaround compiler warning for uninitialized variables, replicated
        // on profile builds on windows with gcc 14.2.0.
        // Also helps with optimizations on some compilers.

        sf_assume(addedSize == 1 || addedSize == 2);
        sf_assume(removedSize == 1 || removedSize == 2);

        if (!(removedSize == 1 || removedSize == 2) || !(addedSize == 1 || addedSize == 2))
            sf_unreachable();

        if ((Forward && removedSize == 1) || (!Forward && addedSize == 1))
        {
            assert(addedSize == 1 && removedSize == 1);
            updateContext.template apply<Add, Sub>(added[0], removed[0]);
        }
        else if (Forward && addedSize == 1)
        {
            assert(removedSize == 2);
            updateContext.template apply<Add, Sub, Sub>(added[0], removed[0], removed[1]);
        }
        else if (!Forward && removedSize == 1)
        {
            assert(addedSize == 2);
            updateContext.template apply<Add, Add, Sub>(added[0], added[1], removed[0]);
        }
        else
        {
            assert(addedSize == 2 && removedSize == 2);
            updateContext.template apply<Add, Add, Sub, Sub>(added[0], added[1], removed[0],
                                                             removed[1]);
        }
    }

    target_state.computed[perspective] = true;
}

void AccumulatorStack::update_accumulator_refresh_cache(Color                            perspective,
                                                        const Position&                  pos,
                                                        AccumulatorState<PSQFeatureSet>& accumulator,
                                                        AccumulatorCaches&               cache) {
    constexpr auto Dimensions = L1;

    using Tiling [[maybe_unused]] = SIMDTiling<Dimensions, Dimensions, PSQTBuckets>;

    const Square             ksq   = pos.square<KING>(perspective);
    auto&                    entry = cache[ksq][perspective];
    PSQFeatureSet::IndexList removed, added;

    const Bitboard changedBB = get_changed_pieces(entry.pieces, pos.piece_array());
    Bitboard       removedBB = changedBB & entry.pieceBB;
    Bitboard       addedBB   = changedBB & pos.pieces();

#if defined(USE_AVX512ICL)
    PSQFeatureSet::write_indices(entry.pieces, pos.piece_array(), removedBB, addedBB, perspective,
                                 ksq, removed, added);
#else
    while (removedBB)
    {
        Square sq = pop_lsb(removedBB);
        removed.push_back(PSQFeatureSet::make_index(perspective, sq, entry.pieces[sq], ksq));
    }
    while (addedBB)
    {
        Square sq = pop_lsb(addedBB);
        added.push_back(PSQFeatureSet::make_index(perspective, sq, pos.piece_on(sq), ksq));
    }
#endif

    entry.pieceBB = pos.pieces();
    entry.pieces  = pos.piece_array();

    accumulator.computed[perspective] = true;

#ifdef VECTOR
    vec_t      acc[Tiling::NumRegs];
    psqt_vec_t psqt[Tiling::NumPsqtRegs];

    const auto* weights = &ft.weights[0];

    for (IndexType j = 0; j < Dimensions / Tiling::TileHeight; ++j)
    {
        auto* accTile =
          reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][j * Tiling::TileHeight]);
        auto* entryTile = reinterpret_cast<vec_t*>(&entry.accumulation[j * Tiling::TileHeight]);

        for (IndexType k = 0; k < Tiling::NumRegs; ++k)
            acc[k] = entryTile[k];

        for (int i = 0; i < removed.ssize(); ++i)
        {
            size_t       index  = removed[i];
            const size_t offset = Dimensions * index;
            auto*        column = reinterpret_cast<const vec_t*>(&weights[offset]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = vec_sub_16(acc[k], column[k]);
        }
        for (int i = 0; i < added.ssize(); ++i)
        {
            size_t       index  = added[i];
            const size_t offset = Dimensions * index;
            auto*        column = reinterpret_cast<const vec_t*>(&weights[offset]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = vec_add_16(acc[k], column[k]);
        }

        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&entryTile[k], acc[k]);
        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&accTile[k], acc[k]);

        weights += Tiling::TileHeight;
    }

    for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
    {
        auto* accTilePsqt = reinterpret_cast<psqt_vec_t*>(
          &accumulator.psqtAccumulation[perspective][j * Tiling::PsqtTileHeight]);
        auto* entryTilePsqt =
          reinterpret_cast<psqt_vec_t*>(&entry.psqtAccumulation[j * Tiling::PsqtTileHeight]);

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            psqt[k] = entryTilePsqt[k];

        for (int i = 0; i < removed.ssize(); ++i)
        {
            size_t       index  = removed[i];
            const size_t offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
            auto*        columnPsqt =
              reinterpret_cast<const psqt_vec_t*>(&ft.psqtWeights[offset]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqt[k] = vec_sub_psqt_32(psqt[k], columnPsqt[k]);
        }
        for (int i = 0; i < added.ssize(); ++i)
        {
            size_t       index  = added[i];
            const size_t offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
            auto*        columnPsqt =
              reinterpret_cast<const psqt_vec_t*>(&ft.psqtWeights[offset]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
        }

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            vec_store_psqt(&entryTilePsqt[k], psqt[k]);
        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            vec_store_psqt(&accTilePsqt[k], psqt[k]);
    }

#else

    for (const auto index : removed)
    {
        const IndexType offset = Dimensions * index;
        for (IndexType j = 0; j < Dimensions; ++j)
            entry.accumulation[j] -= ft.weights[offset + j];

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            entry.psqtAccumulation[k] -= ft.psqtWeights[index * PSQTBuckets + k];
    }
    for (const auto index : added)
    {
        const IndexType offset = Dimensions * index;
        for (IndexType j = 0; j < Dimensions; ++j)
            entry.accumulation[j] += ft.weights[offset + j];

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            entry.psqtAccumulation[k] += ft.psqtWeights[index * PSQTBuckets + k];
    }

    // The accumulator of the refresh entry has been updated.
    // Now copy its content to the actual accumulator we were refreshing.
    accumulator.accumulation[perspective]     = entry.accumulation;
    accumulator.psqtAccumulation[perspective] = entry.psqtAccumulation;
#endif
}

void AccumulatorStack::update_threats_accumulator_full(Color                               perspective,
                                                       const Position&                     pos,
                                                       AccumulatorState<ThreatFeatureSet>& accumulator) {
    constexpr IndexType Dimensions = L1;
    using Tiling [[maybe_unused]]  = SIMDTiling<Dimensions, Dimensions, PSQTBuckets>;

    ThreatFeatureSet::IndexList active;
    ThreatFeatureSet::append_active_indices(perspective, pos, active);

    accumulator.computed[perspective] = true;

#ifdef VECTOR
    vec_t      acc[Tiling::NumRegs];
    psqt_vec_t psqt[Tiling::NumPsqtRegs];

    const auto* threatWeights = &ft.threatWeights[0];

    for (IndexType j = 0; j < Dimensions / Tiling::TileHeight; ++j)
    {
        auto* accTile =
          reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][j * Tiling::TileHeight]);

        for (IndexType k = 0; k < Tiling::NumRegs; ++k)
            acc[k] = vec_zero();

        int i = 0;

        for (; i < active.ssize(); ++i)
        {
            size_t       index  = active[i];
            const size_t offset = Dimensions * index;
            auto*        column = reinterpret_cast<const vec_i8_t*>(&threatWeights[offset]);

    #ifdef USE_NEON
            for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
            {
                acc[k]     = vaddw_s8(acc[k], vget_low_s8(column[k / 2]));
                acc[k + 1] = vaddw_high_s8(acc[k + 1], column[k / 2]);
            }
    #else
            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = vec_add_16(acc[k], vec_convert_8_16(column[k]));
    #endif
        }

        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&accTile[k], acc[k]);

        threatWeights += Tiling::TileHeight;
    }

    for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
    {
        auto* accTilePsqt = reinterpret_cast<psqt_vec_t*>(
          &accumulator.psqtAccumulation[perspective][j * Tiling::PsqtTileHeight]);

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            psqt[k] = vec_zero_psqt();

        for (int i = 0; i < active.ssize(); ++i)
        {
            size_t       index  = active[i];
            const size_t offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
            auto*        columnPsqt =
              reinterpret_cast<const psqt_vec_t*>(&ft.threatPsqtWeights[offset]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
        }

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            vec_store_psqt(&accTilePsqt[k], psqt[k]);
    }

#else

    for (IndexType j = 0; j < Dimensions; ++j)
        accumulator.accumulation[perspective][j] = 0;

    for (std::size_t k = 0; k < PSQTBuckets; ++k)
        accumulator.psqtAccumulation[perspective][k] = 0;

    for (const auto index : active)
    {
        const IndexType offset = Dimensions * index;

        for (IndexType j = 0; j < Dimensions; ++j)
            accumulator.accumulation[perspective][j] +=
              ft.threatWeights[offset + j];

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            accumulator.psqtAccumulation[perspective][k] +=
              ft.threatPsqtWeights[index * PSQTBuckets + k];
    }

#endif
}

// Convert input features
std::int32_t AccumulatorStack::transform(const Position&    pos,
                                         AccumulatorCaches& cache,
                                         TransformedFeatureType*        output,
                                         int                bucket) {

    evaluate_side<PSQFeatureSet>(WHITE, pos, cache);
    evaluate_side<PSQFeatureSet>(BLACK, pos, cache);

    evaluate_side<ThreatFeatureSet>(WHITE, pos, cache);
    evaluate_side<ThreatFeatureSet>(BLACK, pos, cache);

    const auto& accumulatorState       = latest<PSQFeatureSet>();
    const auto& threatAccumulatorState = latest<ThreatFeatureSet>();

    const Color perspectives[2]  = {pos.side_to_move(), ~pos.side_to_move()};
    const auto& psqtAccumulation = accumulatorState.psqtAccumulation;
    auto        psqt =
      (psqtAccumulation[perspectives[0]][bucket] - psqtAccumulation[perspectives[1]][bucket]);

    const auto& threatPsqtAcc = threatAccumulatorState.psqtAccumulation;
    psqt += threatPsqtAcc[perspectives[0]][bucket] - threatPsqtAcc[perspectives[1]][bucket];
    psqt /= 2;

    const auto& accumulation       = accumulatorState.accumulation;
    const auto& threatAccumulation = threatAccumulatorState.accumulation;

    for (IndexType p = 0; p < 2; ++p)
    {
        const IndexType offset = (L1 / 2) * p;

#if defined(VECTOR)

        constexpr IndexType OutputChunkSize = MaxChunkSize;
        static_assert((L1 / 2) % OutputChunkSize == 0);
        constexpr IndexType NumOutputChunks = L1 / 2 / OutputChunkSize;

        const vec_t Zero = vec_zero();
        const vec_t One  = vec_set_16(255);

        const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0]));
        const vec_t* in1 =
          reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][L1 / 2]));
        vec_t* out = reinterpret_cast<vec_t*>(output + offset);

        const vec_t* tin0 =
          reinterpret_cast<const vec_t*>(&(threatAccumulation[perspectives[p]][0]));
        const vec_t* tin1 = reinterpret_cast<const vec_t*>(
          &(threatAccumulation[perspectives[p]][L1 / 2]));

        // Per the NNUE architecture, here we want to multiply pairs of
        // clipped elements and divide the product by 128. To do this,
        // we can naively perform min/max operation to clip each of the
        // four int16 vectors, mullo pairs together, then pack them into
        // one int8 vector. However, there exists a faster way.

        // The idea here is to use the implicit clipping from packus to
        // save us two vec_max_16 instructions. This clipping works due
        // to the fact that any int16 integer below zero will be zeroed
        // on packus.

        // Consider the case where the second element is negative.
        // If we do standard clipping, that element will be zero, which
        // means our pairwise product is zero. If we perform packus and
        // remove the lower-side clip for the second element, then our
        // product before packus will be negative, and is zeroed on pack.
        // The two operation produce equivalent results, but the second
        // one (using packus) saves one max operation per pair.

        // But here we run into a problem: mullo does not preserve the
        // sign of the multiplication. We can get around this by doing
        // mulhi, which keeps the sign. But that requires an additional
        // tweak.

        // mulhi cuts off the last 16 bits of the resulting product,
        // which is the same as performing a rightward shift of 16 bits.
        // We can use this to our advantage. Recall that we want to
        // divide the final product by 128, which is equivalent to a
        // 7-bit right shift. Intuitively, if we shift the clipped
        // value left by 9, and perform mulhi, which shifts the product
        // right by 16 bits, then we will net a right shift of 7 bits.
        // However, this won't work as intended. Since we clip the
        // values to have a maximum value of 127, shifting it by 9 bits
        // might occupy the signed bit, resulting in some positive
        // values being interpreted as negative after the shift.

        // There is a way, however, to get around this limitation. When
        // loading the network, scale accumulator weights and biases by
        // 2. To get the same pairwise multiplication result as before,
        // we need to divide the product by 128 * 2 * 2 = 512, which
        // amounts to a right shift of 9 bits. So now we only have to
        // shift left by 7 bits, perform mulhi (shifts right by 16 bits)
        // and net a 9 bit right shift. Since we scaled everything by
        // two, the values are clipped at 127 * 2 = 254, which occupies
        // 8 bits. Shifting it by 7 bits left will no longer occupy the
        // signed bit, so we are safe.

        // Note that on NEON processors, we shift left by 6 instead
        // because the instruction "vqdmulhq_s16" also doubles the
        // return value after the multiplication, adding an extra shift
        // to the left by 1, so we compensate by shifting less before
        // the multiplication.

        constexpr int shift =
#if defined(USE_SSE2)
          7;
#else
          6;
#endif

        for (IndexType j = 0; j < NumOutputChunks; ++j)
        {
            const vec_t acc0a = vec_add_16(in0[j * 2 + 0], tin0[j * 2 + 0]);
            const vec_t acc0b = vec_add_16(in0[j * 2 + 1], tin0[j * 2 + 1]);
            const vec_t acc1a = vec_add_16(in1[j * 2 + 0], tin1[j * 2 + 0]);
            const vec_t acc1b = vec_add_16(in1[j * 2 + 1], tin1[j * 2 + 1]);

            const vec_t sum0a = vec_slli_16(vec_max_16(vec_min_16(acc0a, One), Zero), shift);
            const vec_t sum0b = vec_slli_16(vec_max_16(vec_min_16(acc0b, One), Zero), shift);
            const vec_t sum1a = vec_min_16(acc1a, One);
            const vec_t sum1b = vec_min_16(acc1b, One);

            const vec_t pa = vec_mulhi_16(sum0a, sum1a);
            const vec_t pb = vec_mulhi_16(sum0b, sum1b);

            out[j] = vec_packus_16(pa, pb);
        }

#else

        for (IndexType j = 0; j < L1 / 2; ++j)
        {
            BiasType sum0 = accumulation[static_cast<int>(perspectives[p])][j + 0];
            BiasType sum1 =
              accumulation[static_cast<int>(perspectives[p])][j + L1 / 2];

            sum0 += threatAccumulation[static_cast<int>(perspectives[p])][j + 0];
            sum1 +=
              threatAccumulation[static_cast<int>(perspectives[p])][j + L1 / 2];

            sum0 = std::clamp<BiasType>(sum0, 0, 255);
            sum1 = std::clamp<BiasType>(sum1, 0, 255);

            output[offset + j] = static_cast<TransformedFeatureType>(unsigned(sum0 * sum1) / 512);
        }

#endif
    }

    return psqt;
}  // end of function transform()

namespace {

template<typename VectorWrapper,
         IndexType Width,
         UpdateOperation... ops,
         typename ElementType,
         typename... Ts,
         std::enable_if_t<is_all_same_v<ElementType, Ts...>, bool> = true>
void fused_row_reduce(const ElementType* in, ElementType* out, const Ts* const... rows) {
    constexpr IndexType size = Width * sizeof(ElementType) / sizeof(typename VectorWrapper::type);

    auto* vecIn  = reinterpret_cast<const typename VectorWrapper::type*>(in);
    auto* vecOut = reinterpret_cast<typename VectorWrapper::type*>(out);

    for (IndexType i = 0; i < size; ++i)
        vecOut[i] = fused<VectorWrapper, ops...>(
          vecIn[i], reinterpret_cast<const typename VectorWrapper::type*>(rows)[i]...);
}

template<typename FeatureSet>
struct AccumulatorUpdateContext {
    Color                               perspective;
    const FeatureTransformer&           featureTransformer;
    const AccumulatorState<FeatureSet>& from;
    AccumulatorState<FeatureSet>&       to;

    AccumulatorUpdateContext(Color                               persp,
                             const FeatureTransformer&           ft,
                             const AccumulatorState<FeatureSet>& accF,
                             AccumulatorState<FeatureSet>&       accT) noexcept :
        perspective{persp},
        featureTransformer{ft},
        from{accF},
        to{accT} {}

    template<UpdateOperation... ops,
             typename... Ts,
             std::enable_if_t<is_all_same_v<IndexType, Ts...>, bool> = true>
    sf_always_inline inline void apply(const Ts... indices) {
        constexpr IndexType Dimensions = L1;

        auto to_weight_vector = [&](const IndexType index) {
            return &featureTransformer.weights[index * Dimensions];
        };

        auto to_psqt_weight_vector = [&](const IndexType index) {
            return &featureTransformer.psqtWeights[index * PSQTBuckets];
        };

        fused_row_reduce<Vec16Wrapper, Dimensions, ops...>(from.accumulation[perspective].data(),
                                                           to.accumulation[perspective].data(),
                                                           to_weight_vector(indices)...);

        fused_row_reduce<Vec32Wrapper, PSQTBuckets, ops...>(
          from.psqtAccumulation[perspective].data(), to.psqtAccumulation[perspective].data(),
          to_psqt_weight_vector(indices)...);
    }

    sf_always_inline inline void apply(const typename FeatureSet::IndexList& added,
               const typename FeatureSet::IndexList& removed) {
        constexpr IndexType Dimensions = L1;

        const auto& fromAcc = from.accumulation[perspective];
        auto&       toAcc   = to.accumulation[perspective];

        const auto& fromPsqtAcc = from.psqtAccumulation[perspective];
        auto&       toPsqtAcc   = to.psqtAccumulation[perspective];

#ifdef VECTOR
        using Tiling = SIMDTiling<Dimensions, Dimensions, PSQTBuckets>;

        vec_t      acc[Tiling::NumRegs];
        psqt_vec_t psqt[Tiling::NumPsqtRegs];

        const auto* threatWeights = &featureTransformer.threatWeights[0];

        for (IndexType j = 0; j < Dimensions / Tiling::TileHeight; ++j)
        {
            auto* fromTile = reinterpret_cast<const vec_t*>(&fromAcc[j * Tiling::TileHeight]);
            auto* toTile   = reinterpret_cast<vec_t*>(&toAcc[j * Tiling::TileHeight]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = fromTile[k];

            for (int i = 0; i < removed.ssize(); ++i)
            {
                size_t       index  = removed[i];
                const size_t offset = Dimensions * index;
                auto*        column = reinterpret_cast<const vec_i8_t*>(&threatWeights[offset]);

    #ifdef USE_NEON
                for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
                {
                    acc[k]     = vsubw_s8(acc[k], vget_low_s8(column[k / 2]));
                    acc[k + 1] = vsubw_high_s8(acc[k + 1], column[k / 2]);
                }
    #else
                for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                    acc[k] = vec_sub_16(acc[k], vec_convert_8_16(column[k]));
    #endif
            }

            for (int i = 0; i < added.ssize(); ++i)
            {
                size_t       index  = added[i];
                const size_t offset = Dimensions * index;
                auto*        column = reinterpret_cast<const vec_i8_t*>(&threatWeights[offset]);

    #ifdef USE_NEON
                for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
                {
                    acc[k]     = vaddw_s8(acc[k], vget_low_s8(column[k / 2]));
                    acc[k + 1] = vaddw_high_s8(acc[k + 1], column[k / 2]);
                }
    #else
                for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                    acc[k] = vec_add_16(acc[k], vec_convert_8_16(column[k]));
    #endif
            }

            for (IndexType k = 0; k < Tiling::NumRegs; k++)
                vec_store(&toTile[k], acc[k]);

            threatWeights += Tiling::TileHeight;
        }

        for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
        {
            auto* fromTilePsqt =
              reinterpret_cast<const psqt_vec_t*>(&fromPsqtAcc[j * Tiling::PsqtTileHeight]);
            auto* toTilePsqt =
              reinterpret_cast<psqt_vec_t*>(&toPsqtAcc[j * Tiling::PsqtTileHeight]);

            for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqt[k] = fromTilePsqt[k];

            for (int i = 0; i < removed.ssize(); ++i)
            {
                size_t       index      = removed[i];
                const size_t offset     = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
                auto*        columnPsqt = reinterpret_cast<const psqt_vec_t*>(
                  &featureTransformer.threatPsqtWeights[offset]);

                for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                    psqt[k] = vec_sub_psqt_32(psqt[k], columnPsqt[k]);
            }

            for (int i = 0; i < added.ssize(); ++i)
            {
                size_t       index      = added[i];
                const size_t offset     = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
                auto*        columnPsqt = reinterpret_cast<const psqt_vec_t*>(
                  &featureTransformer.threatPsqtWeights[offset]);

                for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                    psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
            }

            for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
                vec_store_psqt(&toTilePsqt[k], psqt[k]);
        }

#else

        toAcc     = fromAcc;
        toPsqtAcc = fromPsqtAcc;

        for (const auto index : removed)
        {
            const IndexType offset = Dimensions * index;

            for (IndexType j = 0; j < Dimensions; ++j)
                toAcc[j] -= featureTransformer.threatWeights[offset + j];

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                toPsqtAcc[k] -= featureTransformer.threatPsqtWeights[index * PSQTBuckets + k];
        }

        for (const auto index : added)
        {
            const IndexType offset = Dimensions * index;

            for (IndexType j = 0; j < Dimensions; ++j)
                toAcc[j] += featureTransformer.threatWeights[offset + j];

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                toPsqtAcc[k] += featureTransformer.threatPsqtWeights[index * PSQTBuckets + k];
        }

#endif
    }
};

Bitboard get_changed_pieces(const std::array<Piece, SQUARE_NB>& oldPieces,
                            const std::array<Piece, SQUARE_NB>& newPieces) {
#if defined(USE_AVX2)
    static_assert(sizeof(Piece) == 1);
    Bitboard sameBB = 0;

    for (int i = 0; i < 64; i += 32)
    {
        const __m256i old_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&oldPieces[i]));
        const __m256i new_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&newPieces[i]));
        const __m256i cmpEqual        = _mm256_cmpeq_epi8(old_v, new_v);
        const std::uint32_t equalMask = _mm256_movemask_epi8(cmpEqual);
        sameBB |= static_cast<Bitboard>(equalMask) << i;
    }
    return ~sameBB;
#elif defined(USE_NEON)
    uint8x16x4_t old_v = vld4q_u8(reinterpret_cast<const uint8_t*>(oldPieces.data()));
    uint8x16x4_t new_v = vld4q_u8(reinterpret_cast<const uint8_t*>(newPieces.data()));
    auto         cmp   = [=](const int i) { return vceqq_u8(old_v.val[i], new_v.val[i]); };

    uint8x16_t cmp0_1 = vsriq_n_u8(cmp(1), cmp(0), 1);
    uint8x16_t cmp2_3 = vsriq_n_u8(cmp(3), cmp(2), 1);
    uint8x16_t merged = vsriq_n_u8(cmp2_3, cmp0_1, 2);
    merged            = vsriq_n_u8(merged, merged, 4);
    uint8x8_t sameBB  = vshrn_n_u16(vreinterpretq_u16_u8(merged), 4);

    return ~vget_lane_u64(vreinterpret_u64_u8(sameBB), 0);
#else
    Bitboard changed = 0;

    for (Square sq = SQUARE_ZERO; sq < SQUARE_NB; ++sq)
        changed |= static_cast<Bitboard>(oldPieces[sq] != newPieces[sq]) << sq;

    return changed;
#endif
}

}

}
