"""
Oware Alpha-Beta Solver
=======================
Explores the Oware game tree up to 10 full moves (20 plies) using
alpha-beta pruning with a transposition table. Marks every reachable
position as VIABLE or UNVIABLE for the side to move.

Saves results to the database continuously as it goes — safe to stop
and restart at any time.

Usage
-----
    python oware_solver.py              # run with defaults (20 plies)
    python oware_solver.py --depth 10   # quick test (5 full moves)
    python oware_solver.py --stats      # check progress
    python oware_solver.py --query --board "4,4,4,4,4,4,4,4,4,4,4,4,0,0" --player 0
"""

import sqlite3
import argparse
import time
from typing import Optional

INITIAL_BOARD = (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0)
SOUTH, NORTH = 0, 1


def pit_range(player):
    return range(0, 6) if player == SOUTH else range(6, 12)

def opponent_range(player):
    return range(6, 12) if player == SOUTH else range(0, 6)

def seeds_on_opponent_side(board, player):
    return sum(board[i] for i in opponent_range(player))


def legal_moves(board, player):
    my_pits = list(pit_range(player))
    non_empty = [p for p in my_pits if board[p] > 0]
    if not non_empty:
        return []
    opp_seeds = seeds_on_opponent_side(board, player)
    if opp_seeds > 0:
        return non_empty
    feeding = [p for p in non_empty if board[p] >= (12 - p if player == SOUTH else p - 5)]
    return feeding if feeding else non_empty


def apply_move(board, player, pit):
    b = list(board)
    seeds = b[pit]
    b[pit] = 0
    idx = pit
    while seeds > 0:
        idx = (idx + 1) % 12
        if idx == pit:
            idx = (idx + 1) % 12
        b[idx] += 1
        seeds -= 1

    opp_pits = list(opponent_range(player))
    capture_total = 0
    i = idx
    while i in opp_pits and b[i] in (2, 3):
        capture_total += b[i]
        b[i] = 0
        i = (i - 1) % 12

    if capture_total > 0:
        remaining = sum(b[j] for j in opp_pits)
        if remaining == 0:
            # Grand slam — redo without capture
            b = list(board)
            seeds2 = b[pit]
            b[pit] = 0
            idx2 = pit
            while seeds2 > 0:
                idx2 = (idx2 + 1) % 12
                if idx2 == pit:
                    idx2 = (idx2 + 1) % 12
                b[idx2] += 1
                seeds2 -= 1
        else:
            b[12 if player == SOUTH else 13] += capture_total

    return tuple(b)


def is_terminal(board, player):
    if board[12] >= 25 or board[13] >= 25:
        return True
    return len(legal_moves(board, player)) == 0


def terminal_score(board, player):
    s_total = board[12] + sum(board[i] for i in range(0, 6))
    n_total = board[13] + sum(board[i] for i in range(6, 12))
    my = s_total if player == SOUTH else n_total
    opp = n_total if player == SOUTH else s_total
    return 1000 if my > opp else (-1000 if my < opp else 0)


def evaluate(board, player):
    return (board[12 if player == SOUTH else 13] - board[13 if player == SOUTH else 12]) * 10


class Solver:
    def __init__(self, db_path="oware_results.db"):
        self.db_path = db_path
        self.tt = {}
        self.nodes_visited = 0
        self.positions_saved = 0
        self._init_db()
        self._batch = []  # pending DB writes

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                board_key TEXT PRIMARY KEY,
                player    INTEGER,
                score     INTEGER,
                viable    INTEGER,
                depth     INTEGER,
                timestamp REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_viable ON positions(viable)")
        conn.commit()
        conn.close()

    def _board_key(self, board, player):
        return ",".join(map(str, board)) + f"|{player}"

    def _flush_batch(self):
        if not self._batch:
            return
        conn = sqlite3.connect(self.db_path)
        conn.executemany("""
            INSERT OR REPLACE INTO positions
            (board_key, player, score, viable, depth, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, self._batch)
        conn.commit()
        conn.close()
        self.positions_saved += len(self._batch)
        self._batch = []

    def save_position(self, board, player, score, depth):
        viable = 1 if score >= 0 else 0
        key = self._board_key(board, player)
        self._batch.append((key, player, score, viable, depth, time.time()))
        # Write to disk every 200 positions
        if len(self._batch) >= 200:
            self._flush_batch()

    def already_saved(self, board, player):
        key = self._board_key(board, player)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT 1 FROM positions WHERE board_key=?", (key,)
        ).fetchone()
        conn.close()
        return row is not None

    def load_position(self, board, player):
        key = self._board_key(board, player)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT score, viable, depth FROM positions WHERE board_key=?", (key,)
        ).fetchone()
        conn.close()
        if row:
            return {"score": row[0], "viable": row[1], "depth": row[2]}
        return None

    def db_stats(self):
        conn = sqlite3.connect(self.db_path)
        total    = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        viable   = conn.execute("SELECT COUNT(*) FROM positions WHERE viable=1").fetchone()[0]
        unviable = conn.execute("SELECT COUNT(*) FROM positions WHERE viable=0").fetchone()[0]
        conn.close()
        return {"total": total, "viable": viable, "unviable": unviable}

    def alpha_beta(self, board, player, depth, alpha, beta):
        self.nodes_visited += 1
        tt_key = (board, player)
        if tt_key in self.tt:
            tt_depth, tt_score, tt_flag = self.tt[tt_key]
            if tt_depth >= depth:
                if tt_flag == "exact":
                    return tt_score
                if tt_flag == "lower":
                    alpha = max(alpha, tt_score)
                if tt_flag == "upper":
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        if is_terminal(board, player):
            return terminal_score(board, player)
        if depth == 0:
            return evaluate(board, player)

        moves = legal_moves(board, player)
        if not moves:
            return terminal_score(board, player)

        orig_alpha = alpha
        best_score = -10000

        for pit in moves:
            child = apply_move(board, player, pit)
            score = -self.alpha_beta(child, 1 - player, depth - 1, -beta, -alpha)
            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        flag = "upper" if best_score <= orig_alpha else ("lower" if best_score >= beta else "exact")
        self.tt[tt_key] = (depth, best_score, flag)
        return best_score

    def explore(self, max_depth=20):
        print(f"Oware Solver — exploring up to {max_depth} plies ({max_depth // 2} full moves)")
        print(f"Database: {self.db_path}")
        print("Saving results continuously — safe to stop and restart at any time.\n")

        # BFS: explore and analyse each position as we encounter it
        frontier = [(INITIAL_BOARD, SOUTH, 0)]
        seen = {(INITIAL_BOARD, SOUTH)}
        processed = 0
        t0 = time.time()

        while frontier:
            board, player, ply = frontier.pop(0)

            # Analyse and save this position (unless already in DB)
            if not self.already_saved(board, player):
                remaining = max_depth - ply
                score = self.alpha_beta(board, player, remaining, -10000, 10000)
                self.save_position(board, player, score, remaining)

            processed += 1
            if processed % 500 == 0:
                self._flush_batch()  # ensure DB is up to date
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                stats = self.db_stats()
                print(
                    f"  {processed:,} explored | "
                    f"{stats['total']:,} saved | "
                    f"viable {stats['viable']:,} / unviable {stats['unviable']:,} | "
                    f"{rate:.0f} pos/s"
                )

            # Queue children
            if ply < max_depth:
                for pit in legal_moves(board, player):
                    child = apply_move(board, player, pit)
                    key = (child, 1 - player)
                    if key not in seen:
                        seen.add(key)
                        frontier.append((child, 1 - player, ply + 1))

        self._flush_batch()
        elapsed = time.time() - t0
        stats = self.db_stats()
        print(f"\nDone in {elapsed:.1f}s")
        print(f"  Total saved:  {stats['total']:,}")
        print(f"  Viable:       {stats['viable']:,}")
        print(f"  Unviable:     {stats['unviable']:,}")


def parse_board(s):
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 14:
        raise ValueError("Board must have 14 values: 12 pits + 2 scores")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Oware Alpha-Beta Solver")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--db", type=str, default="oware_results.db")
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--board", type=str, default=None)
    parser.add_argument("--player", type=int, default=0)
    args = parser.parse_args()

    solver = Solver(db_path=args.db)

    if args.stats:
        stats = solver.db_stats()
        print(f"Database: {args.db}")
        print(f"  Total positions: {stats['total']:,}")
        print(f"  Viable:          {stats['viable']:,}")
        print(f"  Unviable:        {stats['unviable']:,}")
        return

    if args.query or args.board:
        board = parse_board(args.board) if args.board else INITIAL_BOARD
        player = args.player
        result = solver.load_position(board, player)
        player_name = "South" if player == SOUTH else "North"
        if result:
            viability = "VIABLE" if result["viable"] else "UNVIABLE"
            print(f"Position for {player_name}: {viability} (score={result['score']}, depth={result['depth']})")
        else:
            print("Position not in database yet — running live search...")
            score = solver.alpha_beta(board, player, 10, -10000, 10000)
            viability = "VIABLE" if score >= 0 else "UNVIABLE"
            print(f"Live result for {player_name}: {viability} (score={score})")
        return

    solver.explore(max_depth=args.depth)


if __name__ == "__main__":
    main()
