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

#include "movepick.h"

#include <cassert>
#include <limits>

#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

namespace {

enum Stages {
    // generate main search moves
    MAIN_TT,
    MAIN_TT2,
    CAPTURE_INIT,
    GOOD_CAPTURE,
    QUIET_INIT,
    GOOD_QUIET,
    BAD_CAPTURE,
    BAD_QUIET,

    // generate evasion moves
    EVASION_TT,
    EVASION_TT2,
    EVASION_INIT,
    EVASION,

    // generate probcut moves
    PROBCUT_TT,
    PROBCUT_TT2,
    PROBCUT_INIT,
    PROBCUT,

    // generate qsearch moves
    QSEARCH_TT,
    QSEARCH_TT2,
    QCAPTURE_INIT,
    QCAPTURE
};

// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p          = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
}

}  // namespace


// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search
MovePicker::MovePicker(const Position&              p,
                       Move                         ttm[2],
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const PawnHistory*           ph,
                       int                          pl) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    pawnHistory(ph),
    depth(d),
    ply(pl) {

    if (pos.checkers())
        stage = EVASION_TT;
    else
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT);

    if (ttm[0] && pos.pseudo_legal(ttm[0]))
    {
        if (ttm[1] && pos.pseudo_legal(ttm[1]))
        {
            ttMove = ttm[0];
            ttMove2 = ttm[1];
        }
        else
        {
            ttMove = Move::none();
            ttMove2 = ttm[0];
            stage++;
        }
    } else {
        ttMove = ttMove2 = Move::none();
        stage += 2;
    }
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.
MovePicker::MovePicker(const Position& p, Move ttm[2], int th, const CapturePieceToHistory* cph) :
    pos(p),
    captureHistory(cph),
    threshold(th) {
    assert(!pos.checkers());

    auto check = [this](Move mov) {
        return mov && pos.capture_stage(mov) && pos.pseudo_legal(mov) && pos.see_ge(mov, threshold);
    };
    stage = PROBCUT_TT;
    if (check(ttm[0]))
    {
        if (check(ttm[1]))
        {
            ttMove = ttm[0];
            ttMove2 = ttm[1];
        }
        else
        {
            ttMove = Move::none();
            ttMove2 = ttm[0];
            stage++;
        }
    } else {
        if (check(ttm[1]))
        {
            ttMove = Move::none();
            ttMove2 = ttm[1];
            stage++;
        } else {
            ttMove = ttMove2 = Move::none();
            stage += 2;
        }
    }
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.
template<GenType Type>
void MovePicker::score() {

    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

    [[maybe_unused]] Bitboard threatenedByPawn, threatenedByMinor, threatenedByRook,
      threatenedPieces;
    if constexpr (Type == QUIETS)
    {
        Color us = pos.side_to_move();

        threatenedByPawn = pos.attacks_by<PAWN>(~us);
        threatenedByMinor =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatenedByPawn;
        threatenedByRook = pos.attacks_by<ROOK>(~us) | threatenedByMinor;

        // Pieces threatened by pieces of lesser material value
        threatenedPieces = (pos.pieces(us, QUEEN) & threatenedByRook)
                         | (pos.pieces(us, ROOK) & threatenedByMinor)
                         | (pos.pieces(us, KNIGHT, BISHOP) & threatenedByPawn);
    }

    for (auto& m : *this)
        if constexpr (Type == CAPTURES)
            m.value =
              7 * int(PieceValue[pos.piece_on(m.to_sq())])
              + (*captureHistory)[pos.moved_piece(m)][m.to_sq()][type_of(pos.piece_on(m.to_sq()))];

        else if constexpr (Type == QUIETS)
        {
            Piece     pc   = pos.moved_piece(m);
            PieceType pt   = type_of(pc);
            Square    from = m.from_sq();
            Square    to   = m.to_sq();

            // histories
            m.value = 2 * (*mainHistory)[pos.side_to_move()][m.from_to()];
            m.value += 2 * (*pawnHistory)[pawn_structure_index(pos)][pc][to];
            m.value += (*continuationHistory[0])[pc][to];
            m.value += (*continuationHistory[1])[pc][to];
            m.value += (*continuationHistory[2])[pc][to];
            m.value += (*continuationHistory[3])[pc][to];
            m.value += (*continuationHistory[5])[pc][to];

            // bonus for checks
            m.value += bool(pos.check_squares(pt) & to) * 16384;

            // bonus for escaping from capture
            m.value += threatenedPieces & from ? (pt == QUEEN && !(to & threatenedByRook)   ? 51700
                                                  : pt == ROOK && !(to & threatenedByMinor) ? 25600
                                                  : !(to & threatenedByPawn)                ? 14450
                                                                                            : 0)
                                               : 0;

            // malus for putting piece en prise
            m.value -= (pt == QUEEN ? bool(to & threatenedByRook) * 49000
                        : pt == ROOK && bool(to & threatenedByMinor) ? 24335
                                                                     : 0);

            if (ply < LOW_PLY_HISTORY_SIZE)
                m.value += 8 * (*lowPlyHistory)[ply][m.from_to()] / (1 + 2 * ply);
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
                m.value =
                  PieceValue[pos.piece_on(m.to_sq())] - type_of(pos.moved_piece(m)) + (1 << 28);
            else
                m.value = (*mainHistory)[pos.side_to_move()][m.from_to()]
                        + (*continuationHistory[0])[pos.moved_piece(m)][m.to_sq()]
                        + (*pawnHistory)[pawn_structure_index(pos)][pos.moved_piece(m)][m.to_sq()];
        }
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.
template<typename Pred>
Move MovePicker::select(Pred filter) {

    for (; cur < endMoves; ++cur)
        if (*cur != ttMove && *cur != ttMove2 && filter())
            return *cur++;

    return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new pseudo-legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.
Move MovePicker::next_move() {

    auto quiet_threshold = [](Depth d) { return -3560 * d; };

top:
    switch (stage)
    {

    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    case MAIN_TT2 :
    case EVASION_TT2 :
    case QSEARCH_TT2 :
    case PROBCUT_TT2 :
        ++stage;
        return ttMove2;

    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT :
        cur = endBadCaptures = moves;
        endMoves             = generate<CAPTURES>(pos, cur);

        score<CAPTURES>();
        partial_insertion_sort(cur, endMoves, std::numeric_limits<int>::min());
        ++stage;
        goto top;

    case GOOD_CAPTURE :
        if (select([&]() {
                // Move losing capture to endBadCaptures to be tried later
                return pos.see_ge(*cur, -cur->value / 18) ? true
                                                          : (*endBadCaptures++ = *cur, false);
            }))
            return *(cur - 1);

        ++stage;
        [[fallthrough]];

    case QUIET_INIT :
        if (!skipQuiets)
        {
            cur      = endBadCaptures;
            endMoves = beginBadQuiets = endBadQuiets = generate<QUIETS>(pos, cur);

            score<QUIETS>();
            partial_insertion_sort(cur, endMoves, quiet_threshold(depth));
        }

        ++stage;
        [[fallthrough]];

    case GOOD_QUIET :
        if (!skipQuiets && select([]() { return true; }))
        {
            if ((cur - 1)->value > -7998 || (cur - 1)->value <= quiet_threshold(depth))
                return *(cur - 1);

            // Remaining quiets are bad
            beginBadQuiets = cur - 1;
        }

        // Prepare the pointers to loop over the bad captures
        cur      = moves;
        endMoves = endBadCaptures;

        ++stage;
        [[fallthrough]];

    case BAD_CAPTURE :
        if (select([]() { return true; }))
            return *(cur - 1);

        // Prepare the pointers to loop over the bad quiets
        cur      = beginBadQuiets;
        endMoves = endBadQuiets;

        ++stage;
        [[fallthrough]];

    case BAD_QUIET :
        if (!skipQuiets)
            return select([]() { return true; });

        return Move::none();

    case EVASION_INIT :
        cur      = moves;
        endMoves = generate<EVASIONS>(pos, cur);

        score<EVASIONS>();
        partial_insertion_sort(cur, endMoves, std::numeric_limits<int>::min());
        ++stage;
        [[fallthrough]];

    case EVASION :
    case QCAPTURE :
        return select([]() { return true; });

    case PROBCUT :
        return select([&]() { return pos.see_ge(*cur, threshold); });
    }

    assert(false);
    return Move::none();  // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }

}  // namespace Stockfish
