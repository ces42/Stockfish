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

// Constants used in NNUE evaluation function

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "../misc.h"

#if defined(USE_AVX2)
    #include <immintrin.h>

#elif defined(USE_SSE41)
    #include <smmintrin.h>

#elif defined(USE_SSSE3)
    #include <tmmintrin.h>

#elif defined(USE_SSE2)
    #include <emmintrin.h>

#elif defined(USE_LASX)
    #include <lasxintrin.h>
    #include <lsxintrin.h>

#elif defined(USE_LSX)
    #include <lsxintrin.h>

#elif defined(USE_NEON)
    #include <arm_neon.h>
#endif

namespace Stockfish::Eval::NNUE {

using BiasType         = i16;
using ThreatWeightType = i8;
using WeightType       = i16;
using PSQTWeightType   = i32;
using IndexType        = u32;

// Version of the evaluation file
constexpr u32 Version = 0x6A448AFAu;

// Constant used in evaluation value calculation
constexpr int OutputScale     = 16;
constexpr int WeightScaleBits = 6;
constexpr int FtMaxVal        = 255;
constexpr int HiddenOneVal    = 128;

// Size of cache line (in bytes)
constexpr usize CacheLineSize = 64;

constexpr const char  Leb128MagicString[]   = "COMPRESSED_LEB128";
constexpr const usize Leb128MagicStringSize = sizeof(Leb128MagicString) - 1;

// SIMD width (in bytes)
#if defined(USE_AVX2)
constexpr usize SimdWidth = 32;

#elif defined(USE_LASX)
constexpr usize SimdWidth = 32;

#elif defined(USE_SSE2)
constexpr usize SimdWidth = 16;

#elif defined(USE_NEON)
constexpr usize SimdWidth = 16;

#elif defined(USE_LSX)
constexpr usize SimdWidth = 16;
#endif

constexpr usize MaxSimdWidth = 32;

#if defined(USE_AVX2) || defined(USE_SSE41)

// Compact candidate values whose preceding byte did not have its continuation bit set.
alignas(CacheLineSize) inline constexpr auto Leb128Shuffle = []() {
    std::array<std::array<u8, 16>, 128> table{};
    for (u32 mask = 0; mask < table.size(); ++mask)
    {
        u32 output = 0;
        for (u32 byte = 0; byte < 8; ++byte)
            if (byte == 0 || !(mask & (u32(1) << (byte - 1))))
            {
                table[mask][2 * output]     = u8(2 * byte);
                table[mask][2 * output + 1] = u8(2 * byte + 1);
                ++output;
            }
        if (mask)
            table[mask][15] = u8(output);
    }
    return table;
}();

#endif

// Type of input feature after conversion
using TransformedFeatureType = u8;

// Round n up to be a multiple of base
template<typename IntType>
constexpr IntType ceil_to_multiple(IntType n, IntType base) {
    return (n + base - 1) / base * base;
}


// Utility to read an integer (signed or unsigned, any size)
// from a stream in little-endian order. We swap the byte order after the read if
// necessary to return a result with the byte ordering of the compiling machine.
template<typename IntType>
inline IntType read_little_endian(std::istream& stream) {
    IntType result;

    if (IsLittleEndian)
        stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
    else
    {
        u8                            u[sizeof(IntType)];
        std::make_unsigned_t<IntType> v = 0;

        stream.read(reinterpret_cast<char*>(u), sizeof(IntType));
        for (usize i = 0; i < sizeof(IntType); ++i)
            v = (v << 8) | u[sizeof(IntType) - i - 1];

        std::memcpy(&result, &v, sizeof(IntType));
    }

    return result;
}


// Utility to write an integer (signed or unsigned, any size)
// to a stream in little-endian order. We swap the byte order before the write if
// necessary to always write in little-endian order, independently of the byte
// ordering of the compiling machine.
template<typename IntType>
inline void write_little_endian(std::ostream& stream, IntType value) {

    if (IsLittleEndian)
        stream.write(reinterpret_cast<const char*>(&value), sizeof(IntType));
    else
    {
        u8                            u[sizeof(IntType)];
        std::make_unsigned_t<IntType> v = value;

        usize i = 0;
        // if constexpr to silence the warning about shift by 8
        if constexpr (sizeof(IntType) > 1)
        {
            for (; i + 1 < sizeof(IntType); ++i)
            {
                u[i] = u8(v);
                v >>= 8;
            }
        }
        u[i] = u8(v);

        stream.write(reinterpret_cast<char*>(u), sizeof(IntType));
    }
}


// Read integers in bulk from a little-endian stream.
// This reads N integers from stream s and puts them in array out.
template<typename IntType>
inline void read_little_endian(std::istream& stream, IntType* out, usize count) {
    if (IsLittleEndian)
        stream.read(reinterpret_cast<char*>(out), sizeof(IntType) * count);
    else
        for (usize i = 0; i < count; ++i)
            out[i] = read_little_endian<IntType>(stream);
}


// Write integers in bulk to a little-endian stream.
// This takes N integers from array values and writes them on stream s.
template<typename IntType>
inline void write_little_endian(std::ostream& stream, const IntType* values, usize count) {
    if (IsLittleEndian)
        stream.write(reinterpret_cast<const char*>(values), sizeof(IntType) * count);
    else
        for (usize i = 0; i < count; ++i)
            write_little_endian<IntType>(stream, values[i]);
}

// Read N signed integers from the stream s, putting them in the array out.
// The stream is assumed to be compressed using the signed LEB128 format.
// See https://en.wikipedia.org/wiki/LEB128 for a description of the compression scheme.
template<typename BufType, typename IntType>
inline void read_leb_128_detail(std::istream& stream,
                                IntType*      out,
                                usize         Count,
                                u32&          bytes_left,
                                BufType&      buf,
                                u32&          buf_pos,
                                u32&          buf_end) {

    static_assert(std::is_signed_v<IntType>, "Not implemented for unsigned types");
    static_assert(sizeof(IntType) <= 4, "Not implemented for types larger than 32 bit");

    u32   result = 0;
    usize shift = 0, i = 0;
    while (i < Count)
    {
        if (buf_pos == buf_end)
        {
            stream.read(reinterpret_cast<char*>(buf.data()),
                        std::min(usize(bytes_left), buf.size()));
            buf_pos = 0;
            buf_end = u32(stream.gcount());

            if (buf_end == 0)
            {
                stream.setstate(std::ios::failbit);
                return;
            }
        }

#if defined(USE_AVX2) || defined(USE_SSE41)
        if constexpr (std::is_same_v<IntType, i16>)
        {
            if (shift == 0 && Count - i >= 8 && buf_end - buf_pos >= 16)
            {
                const __m128i bytes =
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(&buf[buf_pos]));
                const u32 continuationMask = u32(_mm_movemask_epi8(bytes));

                // Consecutive continuation bits denote an unsupported three-byte value.
                if (!(continuationMask & (continuationMask >> 1) & 0xFF))
                {
                    const u32     mask        = continuationMask & 0x7F;
                    const u32     outputCount = mask ? Leb128Shuffle[mask][15] : 8;
                    const u32     consumed    = 8 + ((continuationMask >> 7) & 1);
                    const __m128i payload     = _mm_and_si128(bytes, _mm_set1_epi8(0x7F));
                    const __m128i wide        = _mm_cvtepi8_epi16(bytes);
                    const __m128i values      = _mm_or_si128(
                      _mm_cvtepu8_epi16(payload),
                      _mm_and_si128(
                        wide, _mm_slli_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(payload, 1)), 7)));
                    const __m128i sign =
                      _mm_blendv_epi8(_mm_set1_epi16(0x40), _mm_set1_epi16(0x2000), wide);
                    const __m128i decoded = _mm_sub_epi16(_mm_xor_si128(values, sign), sign);
                    const __m128i shuffle =
                      _mm_load_si128(reinterpret_cast<const __m128i*>(Leb128Shuffle[mask].data()));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]),
                                     _mm_shuffle_epi8(decoded, shuffle));

                    i += outputCount;
                    buf_pos += consumed;
                    bytes_left -= consumed;
                    continue;
                }
            }
        }
#endif

        u8 byte = buf[buf_pos++];
        --bytes_left;
        result |= u32(byte & 0x7f) << (shift % 32);
        shift += 7;

        if ((byte & 0x80) == 0)
        {
            out[i++] = IntType(
              (shift >= 32 || (byte & 0x40) == 0) ? result : result | ~((u32(1) << shift) - 1));
            result = 0;
            shift  = 0;
        }
    }
}

template<typename... Arrays>
inline void read_leb_128(std::istream& stream, Arrays&... outs) {
    // Check the presence of our LEB128 magic string
    char leb128MagicString[Leb128MagicStringSize];
    stream.read(leb128MagicString, Leb128MagicStringSize);
    if (stream.gcount() != Leb128MagicStringSize
        || strncmp(Leb128MagicString, leb128MagicString, Leb128MagicStringSize) != 0)
    {
        stream.setstate(std::ios::failbit);
        return;
    }

    auto                 bytes_left = read_little_endian<u32>(stream);
    std::array<u8, 8192> buf;
    u32                  buf_pos = 0, buf_end = 0;

    (read_leb_128_detail(stream, outs.data(), outs.size(), bytes_left, buf, buf_pos, buf_end), ...);

    if (bytes_left != 0)
        stream.setstate(std::ios::failbit);
}


template<typename IntType>
inline void read_leb_128(std::istream& stream, IntType* out, usize expected) {
    // Check the presence of our LEB128 magic string
    char leb128MagicString[Leb128MagicStringSize];
    stream.read(leb128MagicString, Leb128MagicStringSize);
    if (stream.gcount() != Leb128MagicStringSize
        || strncmp(Leb128MagicString, leb128MagicString, Leb128MagicStringSize) != 0)
    {
        stream.setstate(std::ios::failbit);
        return;
    }

    auto                 bytes_left = read_little_endian<u32>(stream);
    std::array<u8, 8192> buf;
    u32                  buf_pos = 0, buf_end = 0;

    read_leb_128_detail(stream, out, expected, bytes_left, buf, buf_pos, buf_end);

    if (bytes_left != 0)
        stream.setstate(std::ios::failbit);
}


// Write signed integers to a stream with LEB128 compression.
// This takes N integers from array values, compresses them with
// the LEB128 algorithm and writes the result on the stream s.
// See https://en.wikipedia.org/wiki/LEB128 for a description of the compression scheme.
template<typename IntType>
inline void write_leb_128(std::ostream& stream, const IntType* values, usize Count) {

    // Write our LEB128 magic string
    stream.write(Leb128MagicString, Leb128MagicStringSize);

    static_assert(std::is_signed_v<IntType>, "Not implemented for unsigned types");

    u32 byte_count = 0;
    for (usize i = 0; i < Count; ++i)
    {
        IntType value = values[i];
        u8      byte;
        do
        {
            byte = value & 0x7f;
            value >>= 7;
            ++byte_count;
        } while ((byte & 0x40) == 0 ? value != 0 : value != -1);
    }

    write_little_endian(stream, byte_count);

    const u32 BUF_SIZE = 4096;
    u8        buf[BUF_SIZE];
    u32       buf_pos = 0;

    auto flush = [&]() {
        if (buf_pos > 0)
        {
            stream.write(reinterpret_cast<char*>(buf), buf_pos);
            buf_pos = 0;
        }
    };

    auto write = [&](u8 b) {
        buf[buf_pos++] = b;
        if (buf_pos == BUF_SIZE)
            flush();
    };

    for (usize i = 0; i < Count; ++i)
    {
        IntType value = values[i];
        while (true)
        {
            u8 byte = value & 0x7f;
            value >>= 7;
            if ((byte & 0x40) == 0 ? value == 0 : value == -1)
            {
                write(byte);
                break;
            }
            write(byte | 0x80);
        }
    }

    flush();
}

template<typename IntType, usize Count>
inline void write_leb_128(std::ostream& stream, const std::array<IntType, Count>& values) {
    write_leb_128(stream, values.data(), Count);
}

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_COMMON_H_INCLUDED
