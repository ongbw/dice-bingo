"""
Project: Dice Bingo
Authors: David Hemmer and Benjamin Ong
License: MIT

Copyright (c) 2026 Benjamin Ong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
Compilable Python Dice Bingo that computes the expected
expected number of rolls to completion for a given board



packages used:
numba, a just-in-time compiler for Python
argparse, handling command line options

call:

python dice_bingo.py start_id end_id

code systematically computes the completion time for card_id,
where start_id <= card_id <= end_id

the function decode_card(card_id) converts the card_id (an integer)
into an aray of 9 numbers, each between 2 and 12.

upon completion, it returns the card_id that has the smallest
completion time for the given range.

"""

import numpy as np
from numba import njit
import argparse

# ------------------------------------------------------------
# CONSTANTS (Numba-friendly)
# ------------------------------------------------------------

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

WIN_MASKS = np.array(
    [(1<<a)|(1<<b)|(1<<c) for (a,b,c) in WIN_LINES],
    dtype=np.int64
)

# probabilities for sums 2..12
P_ARRAY = np.array(
    [1,2,3,4,5,6,5,4,3,2,1],
    dtype=np.float64
) / 36.0

## generating cards
@njit
def decode_card(card_id):
    """
    Convert integer → 9 cell values (2..12)
    """
    board = np.empty(9, dtype=np.int64)

    x = card_id
    for i in range(9):
        board[i] = 2 + (x % 11)
        x //= 11

    return board

# define search range
@njit
def search_range(start_id, end_id,
                 tol, max_iter,
                 win_masks, probs):

    best_val = 1e30
    best_id = -1

    for cid in range(start_id, end_id):

        val = evaluate_card(
            cid,
            tol,
            max_iter,
            win_masks,
            probs
        )

        if val < best_val:
            best_val = val
            best_id = cid

    return best_id, best_val

# single card evaluation kernel, given card id
@njit
def evaluate_card(card_id,
                  tol, max_iter,
                  win_masks, probs):

    board = decode_card(card_id)
    label_mask = build_masks_nb(board)

    mask0 = 0  # empty board start

    return expected_rolls_numba(
        mask0,
        label_mask,
        tol,
        max_iter,
        win_masks,
        probs
    )



# ------------------------------------------------------------
# NUMBA FUNCTIONS (HOT LOOP)
# ------------------------------------------------------------

@njit
def is_win_nb(mask, win_masks):
    for i in range(len(win_masks)):
        wm = win_masks[i]
        if (mask & wm) == wm:
            return True
    return False


@njit
def expected_rolls_numba(mask0, label_mask, tol, max_iter,
                         win_masks, probs):

    if is_win_nb(mask0, win_masks):
        return 0.0

    E = np.zeros(512, dtype=np.float64)

    # initial guess
    for m in range(512):
        if not is_win_nb(m, win_masks):
            E[m] = 20.0

    for _ in range(max_iter):

        delta = 0.0

        for m in range(512):

            if is_win_nb(m, win_masks):
                continue

            Em = E[m]
            acc = 0.0

            # loop over sums 2..12
            for si in range(11):

                unmarked = label_mask[si] & (~m) & 0x1FF

                if unmarked == 0:
                    best = Em
                else:
                    best = Em
                    mm = unmarked

                    while mm:
                        lsb = mm & -mm
                        mp = m | lsb
                        v = E[mp]
                        if v < best:
                            best = v
                        mm -= lsb

                acc += probs[si] * best

            vnew = 1.0 + acc
            d = abs(vnew - Em)

            if d > delta:
                delta = d

            E[m] = vnew

        if delta < tol:
            break

    return E[mask0]


# ------------------------------------------------------------
# PYTHON SETUP CODE (NOT PERFORMANCE CRITICAL)
# ------------------------------------------------------------

def parse_position(spec, dummy_label_for_X=7):

    if isinstance(spec, str):
        rows = [r.strip() for r in spec.split("/")]
        grid = []
        for r in rows:
            grid.append([c.strip() for c in r.split(",")])
    else:
        grid = spec

    board = []
    mask = 0
    idx = 0

    for r in range(3):
        for c in range(3):
            token = grid[r][c]
            if isinstance(token, str) and token.lower() == "x":
                mask |= (1 << idx)
                board.append(dummy_label_for_X)
            else:
                board.append(int(token))
            idx += 1

    return tuple(board), mask


@njit
def build_masks_nb(board):
    masks = np.zeros(11, dtype=np.int64)

    for i in range(9):
        v = board[i] - 2
        masks[v] |= (1 << i)

    return masks



def expected_rolls_from_position(spec,
                                 tol=1e-12,
                                 max_iter=200000):

    board, mask0 = parse_position(spec)
    label_mask = build_avail_masks_array(board)

    return expected_rolls_numba(
        mask0,
        label_mask,
        tol,
        max_iter,
        WIN_MASKS,
        P_ARRAY
    )

# ------------------------------------------------------------
# DRIVER
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Search dice bingo cards over an ID range."
    )

    parser.add_argument("start", type=int,
                        help="Starting card id (inclusive)")
    parser.add_argument("end", type=int,
                        help="Ending card id (exclusive)")

    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_iter", type=int, default=200000)

    args = parser.parse_args()

    start = args.start
    end = args.end

    print(f"Searching cards [{start}, {end})")

    best_id, best_val = search_range(
        start, end,
        1e-12, 200000,
        WIN_MASKS,
        P_ARRAY
    )

    print("Best card id:", best_id)
    print("Best value:", best_val)
