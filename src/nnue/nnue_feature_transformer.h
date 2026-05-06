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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <iterator>

#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE {

// Returns the inverse of a permutation
template<std::size_t Len>
constexpr std::array<std::size_t, Len>
invert_permutation(const std::array<std::size_t, Len>& order) {
    std::array<std::size_t, Len> inverse{};
    for (std::size_t i = 0; i < order.size(); i++)
        inverse[order[i]] = i;
    return inverse;
}

// Divide a byte region of size TotalSize to chunks of size
// BlockSize, and permute the blocks by a given order
template<std::size_t BlockSize, typename T, std::size_t N, std::size_t OrderSize>
void permute(std::array<T, N>& data, const std::array<std::size_t, OrderSize>& order) {
    constexpr std::size_t TotalSize = N * sizeof(T);

    static_assert(TotalSize % (BlockSize * OrderSize) == 0,
                  "ChunkSize * OrderSize must perfectly divide TotalSize");

    constexpr std::size_t ProcessChunkSize = BlockSize * OrderSize;

    std::array<std::byte, ProcessChunkSize> buffer{};

    std::byte* const bytes = reinterpret_cast<std::byte*>(data.data());

    for (std::size_t i = 0; i < TotalSize; i += ProcessChunkSize)
    {
        std::byte* const values = &bytes[i];

        for (std::size_t j = 0; j < OrderSize; j++)
        {
            auto* const buffer_chunk = &buffer[j * BlockSize];
            auto* const value_chunk  = &values[order[j] * BlockSize];

            std::copy(value_chunk, value_chunk + BlockSize, buffer_chunk);
        }

        std::copy(std::begin(buffer), std::end(buffer), values);
    }
}

// Input feature converter
class FeatureTransformer {
   public:
    // Number of input/output dimensions
    static constexpr IndexType InputDimensions =
      PSQFeatureSet::Dimensions + ThreatFeatureSet::Dimensions;

    // Size of forward propagation buffer
    static constexpr std::size_t BufferSize = L1 * sizeof(TransformedFeatureType);

    // Store the order by which 128-bit blocks of a 1024-bit data must
    // be permuted so that calling packus on adjacent vectors of 16-bit
    // integers loaded from the data results in the pre-permutation order
    static constexpr auto PackusEpi16Order = []() -> std::array<std::size_t, 8> {
#if defined(USE_AVX512)
        // _mm512_packus_epi16 after permutation:
        // |   0   |   2   |   4   |   6   | // Vector 0
        // |   1   |   3   |   5   |   7   | // Vector 1
        // | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | // Packed Result
        return {0, 2, 4, 6, 1, 3, 5, 7};
#elif defined(USE_AVX2)
        // _mm256_packus_epi16 after permutation:
        // |   0   |   2   |  |   4   |   6   | // Vector 0, 2
        // |   1   |   3   |  |   5   |   7   | // Vector 1, 3
        // | 0 | 1 | 2 | 3 |  | 4 | 5 | 6 | 7 | // Packed Result
        return {0, 2, 1, 3, 4, 6, 5, 7};
#else
        return {0, 1, 2, 3, 4, 5, 6, 7};
#endif
    }();

    static constexpr auto InversePackusEpi16Order = invert_permutation(PackusEpi16Order);

    static constexpr std::uint32_t combine_hash(std::initializer_list<std::uint32_t> hashes) {
        std::uint32_t hash = 0;
        for (const auto component_hash : hashes)
        {
            hash = (hash << 1) | (hash >> 31);
            hash ^= component_hash;
        }
        return hash;
    }

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() {
        return combine_hash({ThreatFeatureSet::HashValue, PSQFeatureSet::HashValue})
             ^ (L1 * 2);
    }

    void permute_weights() {
        permute<16>(biases, PackusEpi16Order);
        permute<16>(weights, PackusEpi16Order);

        permute<8>(threatWeights, PackusEpi16Order);
    }

    void unpermute_weights() {
        permute<16>(biases, InversePackusEpi16Order);
        permute<16>(weights, InversePackusEpi16Order);
        permute<8>(threatWeights, InversePackusEpi16Order);
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_leb_128(stream, biases);

        read_little_endian<ThreatWeightType>(stream, threatWeights.data(),
                                             ThreatFeatureSet::Dimensions * L1);
        read_leb_128(stream, weights);
        read_leb_128(stream, threatPsqtWeights, psqtWeights);

        permute_weights();

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        std::unique_ptr<FeatureTransformer> copy = std::make_unique<FeatureTransformer>(*this);

        copy->unpermute_weights();

        write_leb_128<BiasType>(stream, copy->biases);

        write_little_endian<ThreatWeightType>(stream, copy->threatWeights.data(),
                                              ThreatFeatureSet::Dimensions * L1);
        write_leb_128<WeightType>(stream, copy->weights);

        auto combinedPsqtWeights =
          std::make_unique<std::array<PSQTWeightType, InputDimensions * PSQTBuckets>>();

        std::copy(std::begin(copy->threatPsqtWeights),
                  std::begin(copy->threatPsqtWeights) + ThreatFeatureSet::Dimensions * PSQTBuckets,
                  combinedPsqtWeights->begin());

        std::copy(std::begin(copy->psqtWeights),
                  std::begin(copy->psqtWeights) + PSQFeatureSet::Dimensions * PSQTBuckets,
                  combinedPsqtWeights->begin() + ThreatFeatureSet::Dimensions * PSQTBuckets);

        write_leb_128<PSQTWeightType>(stream, *combinedPsqtWeights);

        return !stream.fail();
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;

        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_raw_data_hash(psqtWeights));

        hash_combine(h, get_raw_data_hash(threatWeights));
        hash_combine(h, get_raw_data_hash(threatPsqtWeights));

        hash_combine(h, get_hash_value());

        return h;
    }

    alignas(CacheLineSize) std::array<BiasType, L1> biases;
    alignas(
      CacheLineSize) std::array<WeightType, L1 * PSQFeatureSet::Dimensions> weights;
    alignas(CacheLineSize)
      std::array<ThreatWeightType, L1 * ThreatFeatureSet::Dimensions> threatWeights;
    alignas(CacheLineSize)
      std::array<PSQTWeightType, PSQFeatureSet::Dimensions * PSQTBuckets> psqtWeights;
    alignas(CacheLineSize)
      std::array<PSQTWeightType, ThreatFeatureSet::Dimensions * PSQTBuckets> threatPsqtWeights;
};

}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::FeatureTransformer> {
    std::size_t operator()(const Stockfish::Eval::NNUE::FeatureTransformer& ft) const noexcept {
        return ft.get_content_hash();
    }
};

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
