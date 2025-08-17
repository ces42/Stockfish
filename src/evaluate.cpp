/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

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

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

// Returns a static, purely materialistic evaluation of the position from
// the point of view of the side to move. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position& pos) {
    Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
         + (pos.non_pawn_material(c) - pos.non_pawn_material(~c));
}

bool Eval::use_smallnet(const Position& pos, bool last_big) {
    // int center_kings = (rank_of(pos.square<KING>(WHITE)) >= 2) + (rank_of(pos.square<KING>(BLACK)) <= 5);
    // dbg_mean_of(pos.count<ALL_PIECES>(), 5);
    // dbg_mean_of(popcount(pos.pinners(BLACK) | pos.pinners(WHITE)), 6);
    return std::abs(simple_eval(pos)) > 900 + 80 * last_big;
}

# define DEV(num, id) dbg_mean_of((num) * (num) / 1000, (id))

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks&    networks,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());

    bool last_big = accumulators.size > 1
        && accumulators.accumulators[accumulators.size - 2].accumulatorBig.computed[BLACK]
        && accumulators.accumulators[accumulators.size - 2].accumulatorBig.computed[WHITE];
    bool smallNet           = use_smallnet(pos, last_big);
    auto [psqt, positional] = smallNet ? networks.small.evaluate(pos, accumulators, &caches.small)
                                       : networks.big.evaluate(pos, accumulators, &caches.big);

    Value nnue = (125 * psqt + 131 * positional) / 128;
    Value nnue_big, nnue_small = nnue;

    // Re-evaluate the position when higher eval accuracy is worth the time spent
    // dbg_hit_on(smallNet && (std::abs(nnue) < 236), 1);
    if (smallNet && (std::abs(nnue) < 236))
    {
        std::tie(psqt, positional) = networks.big.evaluate(pos, accumulators, &caches.big);
        nnue                       = (125 * psqt + 131 * positional) / 128;
        smallNet                   = false;
        nnue_big = nnue;
        // DEV(nnue - nnue_small, 0);
    }
    // else if (smallNet) {
    //     auto [psqt_big, positional_big] = networks.big.evaluate(pos, accumulators, &caches.big);
    //     nnue_big = (125 * psqt_big + 131 * positional_big) / 128;
    //     DEV(nnue_big - nnue, 1);
    // } else {
    //     nnue_big = nnue;
    //     auto [psqt_small, positional_small] = networks.small.evaluate(pos, accumulators, &caches.small);
    //     nnue_small = (125 * psqt_small + 131 * positional_small) / 128;
    //     // DEV(nnue - nnue_small, 0);
    // }
    // dbg_hit_on(smallNet);
    // int diff = std::abs(nnue_big - nnue_small);
    // int diff_rm = diff - int(0.090037 * std::abs(simple_eval(pos)));
    // dbg_correl_of(std::abs(nnue), diff, 8);
    // dbg_correl_of(std::abs(nnue), diff_rm, 7);
    //
    // dbg_correl_of(std::abs(simple_eval(pos)), diff, 0);
    // dbg_correl_of(std::abs(simple_eval(pos)), diff_rm, 9);
    //
    // dbg_correl_of( pos.count<ALL_PIECES>(), diff_rm, 1);
    // dbg_correl_of( pos.count<PAWN>(), diff_rm, 2);
    // dbg_correl_of( pos.count<KNIGHT>(), diff_rm, 3);
    // dbg_correl_of( pos.count<QUEEN>(), diff_rm, 4);
    // dbg_correl_of( popcount(pos.blockers_for_king(WHITE) | pos.blockers_for_king(BLACK)), diff_rm, 5);
    // dbg_correl_of(
    //     (rank_of(pos.square<KING>(WHITE)) >= 2) + (rank_of(pos.square<KING>(BLACK)) <= 5),
    //     diff_rm,
    //     6);

    // Blend optimism and eval with nnue complexity
    int nnueComplexity = std::abs(psqt - positional);
    optimism += optimism * nnueComplexity / 468;
    nnue -= nnue * nnueComplexity / 18000;

    int material = 535 * pos.count<PAWN>() + pos.non_pawn_material();
    int v        = (nnue * (77777 + material) + optimism * (7777 + material)) / 77777;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 212;

    // Guarantee evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Networks& networks) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    Eval::NNUE::AccumulatorStack accumulators;
    auto                         caches = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, networks, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    auto [psqt, positional] = networks.big.evaluate(pos, accumulators, &caches->big);
    Value v                 = psqt + positional;
    v                       = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = evaluate(networks, pos, accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "Final evaluation       " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]";
    ss << "\n";

    return ss.str();
}

}  // namespace Stockfish
