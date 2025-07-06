import pygame
import sys
import time
import random
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pygame.locals import * # QUITなどの定数をインポート

# Color definitions (色の定義)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 204, 0)
DARK_BG = (40, 40, 40)
LIGHT_BG = (230, 230, 230)
GRAY = (180, 180, 180)
LIGHT_GRAY = (200, 200, 200) # 新しい色定義
DARK_GREEN_BOARD = (0, 80, 0) # ダークモード用のボード色

# Board size (ボードのサイズ)
BOARD_SIZE = 8
CELL_SIZE = 60
BOARD_PIXEL = BOARD_SIZE * CELL_SIZE

# Constants for AI Depth (AI深度の定数)
MAX_AI_DEPTH = 15 # 新しいAI深度の最大値

# UI areas (UI領域のサイズ)
TOP_UI = 120
# 画面サイズを拡張するためにBOTTOM_UIを調整
BOTTOM_UI = 342
WINDOW_WIDTH = BOARD_PIXEL + 80 # 左右に40pxずつ余白
WINDOW_HEIGHT = TOP_UI + BOARD_PIXEL + BOTTOM_UI # 新しいBOTTOM_UIを反映

FONT_NAME = None # フォント名 (システムフォントを使用するためNone)


class ReversiGame:
    def __init__(self):
        pygame.init()
        # CPU/スレッド情報の取得とコンソール表示
        self.cpu_count = multiprocessing.cpu_count()
        self.max_workers = self.cpu_count # MCTS用スレッド数
        print(f"利用可能なCPUコア数: {self.cpu_count}")
        print(f"AI思考用スレッド数 (MCTS): {self.max_workers}")
        print(f"AI思考用プロセス数 (AlphaBeta): {self.cpu_count}")

        # AlphaBeta用マルチプロセスプールを初期化
        self.process_pool = multiprocessing.Pool(processes=self.cpu_count)
        
        global FONT_NAME
        # フォントを複数候補で明示指定
        for font_name in ["Arial Unicode MS", "Noto Sans CJK JP", "Meiryo", "Arial", pygame.font.get_default_font()]:
            try:
                self.font = pygame.font.SysFont(font_name, 28)
                self.small_font = pygame.font.SysFont(font_name, 20)
                self.large_font = pygame.font.SysFont(font_name, 36)
                FONT_NAME = font_name
                # テスト描画で英数字が使えるか確認
                test = self.font.render("Test", True, (0,0,0))
                break
            except:
                continue
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('PyReversi') # タイトルをPyReversiに変更
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.board[3][3] = self.board[4][4] = 2  # White
        self.board[3][4] = self.board[4][3] = 1  # Black
        self.current_player = 1  # Black starts
        self.game_over = False
        self.dark_mode = False
        self.ai_depth = 3 # AI Depthのデフォルト値を3に設定
        self.hint_mode = False # Hintモードの状態
        self.undo_stack = []
        self.redo_stack = []
        self.player_types = {1: "Human", 2: "AI"} # プレイヤータイプを辞書で管理 (1:Black, 2:White)
        self.best_move = None # AIが計算した最善手
        self.button_rects = {} # ボタンのRectを保存
        self.slider_drag = False # スライダーのドラッグ状態
        self.slider_rect = None # スライダーのRect
        self.player_btn_rects = [[None, None], [None, None]] # プレイヤータイプボタンのRectを保存
        
        self.ponder_thread = None # 先読みスレッド
        self.ponder_result = None # 先読み結果
        self.ponder_move = None # 先読みの基準となったHumanの最善手
        self.ponder_stop = threading.Event() # 先読み停止シグナル
        
        self.ai_mode = "AlphaBeta" # AIモードのデフォルト値

        # AI評価関数の重み (初期値)
        self.weights = [
            [150, -40, 30, 10, 10, 30, -40, 150],
            [-40, -60, -5, -5, -5, -5, -60, -40],
            [30, -5, 20, 3, 3, 20, -5, 30],
            [10, -5, 3, 3, 3, 3, -5, 10],
            [10, -5, 3, 3, 3, 3, -5, 10],
            [30, -5, 20, 3, 3, 20, -5, 30],
            [-40, -60, -5, -5, -5, -5, -60, -40],
            [150, -40, 30, 10, 10, 30, -40, 150],
        ]

        # 効果音の初期化
        try:
            pygame.mixer.init()
            self.flip_sound = pygame.mixer.Sound("flip.mp3") # flip.mp3 が存在すると仮定
        except Exception as e:
            print(f"[Sound Error] 効果音の初期化に失敗しました: {e}")
            self.flip_sound = None

    @staticmethod
    def _get_valid_moves(board, player):
        """Returns a list of valid moves (x, y) for the given player on the given board."""
        moves = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if ReversiGame._is_valid_move_for(board, player, x, y):
                    moves.append((x, y))
        return moves

    @staticmethod
    def _is_valid_move_for(board, player, x, y):
        """Checks if a move (x, y) is valid for the player on the board."""
        if board[y][x] != 0:
            return False
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dx, dy in directions:
            if ReversiGame._can_flip_for(board, player, x, y, dx, dy):
                return True
        return False

    @staticmethod
    def _can_flip_for(board, player, x, y, dx, dy):
        """Checks if a stone can be flipped in a specific direction from (x, y)."""
        opponent = 3 - player
        x, y = x + dx, y + dy
        found_opponent = False
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            if board[y][x] == opponent:
                found_opponent = True
            elif board[y][x] == player:
                return found_opponent
            else:
                break
            x, y = x + dx, y + dy
        return False

    @staticmethod
    def _simulate_move(board, player, x, y):
        """Applies a move to a board and flips the necessary stones."""
        # This modifies the board in place
        board[y][x] = player
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        opponent = 3 - player
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            stones = []
            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[ny][nx] == opponent:
                stones.append((nx, ny))
                nx += dx
                ny += dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[ny][nx] == player:
                for fx, fy in stones:
                    board[fy][fx] = player

    @staticmethod
    def _count_frontier_discs(board, player):
        """Counts the number of frontier discs for a player (discs adjacent to empty squares)."""
        frontier = 0
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board[y][x] == player:
                    for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                            if board[ny][nx] == 0:
                                frontier += 1
                                break # Found an empty neighbor, move to next disc
        return frontier

    @staticmethod
    def _get_strong_stable_discs(board, player):
        """Identifies and counts strongly stable discs (discs that cannot be flipped)."""
        num_stable = 0
        stable_map = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        # Check corners first
        corners = [(0,0), (0,BOARD_SIZE-1), (BOARD_SIZE-1,0), (BOARD_SIZE-1,BOARD_SIZE-1)]
        for r, c in corners:
            if board[r][c] == player:
                stable_map[r][c] = True
                num_stable += 1
        
        # Spread stability from corners (simplified)
        # This is a basic implementation. A truly robust stable disc calculation is more complex.
        changed = True
        while changed:
            changed = False
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if stable_map[r][c] or board[r][c] != player:
                        continue
                    
                    stable_dirs = 0
                    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
                    for dx, dy in directions:
                        nx, ny = c, r
                        path_stable = False
                        while 0 <= nx+dx < BOARD_SIZE and 0 <= ny+dy < BOARD_SIZE:
                            nx += dx
                            ny += dy
                            if board[ny][nx] != player:
                                break
                            if stable_map[ny][nx]:
                                path_stable = True
                                break
                        if path_stable or not (0 <= nx+dx < BOARD_SIZE and 0 <= ny+dy < BOARD_SIZE):
                            stable_dirs += 1
                    
                    if stable_dirs == 8:
                        stable_map[r][c] = True
                        changed = True
                        num_stable += 1
        
        return num_stable, stable_map

    @staticmethod
    def _evaluate_board(board, player, weights_config):
        """Evaluates the board for the given player using heuristic weights and other factors."""
        opponent = 3 - player
        
        # Dynamic weights based on game stage (determined by empty_count)
        empty_count = sum(row.count(0) for row in board)
        
        # Use the passed weights_config. This allows for tuning.
        # The weights_config is assumed to be a 2D list of weights.
        weights = weights_config
                
        score = 0
        # Positional score
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board[y][x] == player:
                    score += weights[y][x]
                elif board[y][x] == opponent:
                    score -= weights[y][x]
        
        # Disc count (important in endgame)
        my_count = sum(row.count(player) for row in board)
        op_count = sum(row.count(opponent) for row in board)
        score += (my_count - op_count) * 10
        
        # Mobility (number of valid moves)
        my_moves = len(ReversiGame._get_valid_moves(board, player))
        op_moves = len(ReversiGame._get_valid_moves(board, opponent))
        score += (my_moves - op_moves) * 5
        
        # Frontier discs (bad to have many, exposes more pieces)
        my_frontier = ReversiGame._count_frontier_discs(board, player)
        op_frontier = ReversiGame._count_frontier_discs(board, opponent)
        score -= (my_frontier - op_frontier) * 4
        
        # Stable discs (cannot be flipped)
        my_strong_stable, _ = ReversiGame._get_strong_stable_discs(board, player)
        op_strong_stable, _ = ReversiGame._get_strong_stable_discs(board, opponent)
        score += (my_strong_stable - op_strong_stable) * 30
        
        # Corner capture bonus/penalty
        corners = [(0,0), (0,BOARD_SIZE-1), (BOARD_SIZE-1,0), (BOARD_SIZE-1,BOARD_SIZE-1)]
        for r, c in corners:
            if board[r][c] == player:
                score += 100
            elif board[r][c] == opponent:
                score -= 100
        
        # X-squares (bad squares near corners)
        x_squares = [(1,1),(1,6),(6,1),(6,6)]
        for r, c in x_squares:
            if board[r][c] == player:
                score -= 25
            elif board[r][c] == opponent:
                score += 25
                
        # C-squares (bad squares adjacent to corners)
        c_squares = [(0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6)]
        for r, c in c_squares:
            if board[r][c] == player:
                score -= 15
            elif board[r][c] == opponent:
                score += 15

        # A-squares (other bad squares near corners)
        a_squares = [(0,2),(2,0),(0,5),(2,7),(5,0),(7,2),(5,7),(7,5)]
        for r, c in a_squares:
            if board[r][c] == player:
                score -= 8
            elif board[r][c] == opponent:
                score += 8
        
        # Parity (who makes the last move)
        if empty_count % 2 != 0:
            score += 2
        else:
            score -= 2
            
        # Lookahead Mobility (predict future mobility)
        def lookahead_mobility_static(b, p, depth):
            if depth == 0:
                return len(ReversiGame._get_valid_moves(b, p))
            moves = ReversiGame._get_valid_moves(b, p)
            if not moves:
                return 0
            total = 0
            for x, y in moves:
                new_b = [row[:] for row in b]
                ReversiGame._simulate_move(new_b, 3-p, x, y)
                total += lookahead_mobility_static(new_b, p, depth-1)
            return total
        
        score += (lookahead_mobility_static(board, player, 2) - lookahead_mobility_static(board, opponent, 2)) * 1.5
        
        return score

    @staticmethod
    def _quiescence(board, player, alpha, beta, eval_func, weights_config, q_depth_limit=3): # Added q_depth_limit
        """Performs a quiescence search to evaluate noisy positions (those with captures)."""
        
        if q_depth_limit == 0: # Base case for quiescence depth
            return eval_func(board, player, weights_config)

        stand_pat = eval_func(board, player, weights_config)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        moves = ReversiGame._get_valid_moves(board, player)
        
        # Filter moves to only include "noisy" moves (captures).
        # For Reversi, all valid moves involve captures, so this is implicitly handled.
        # If there were non-capture moves, we would filter them out here.

        for x, y in moves:
            new_board = [row[:] for row in board]
            ReversiGame._simulate_move(new_board, player, x, y)
            
            score = -ReversiGame._quiescence(new_board, 3 - player, -beta, -alpha, eval_func, weights_config, q_depth_limit - 1) # Decrement q_depth_limit
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    @staticmethod
    def _board_hash(board, player):
        """Generates a unique hash for the board state and player, considering symmetries."""
        boards = [
            tuple(tuple(row) for row in board),
            tuple(tuple(row[::-1]) for row in board),
            tuple(tuple(board[y]) for y in range(BOARD_SIZE-1, -1, -1)),
            tuple(tuple(board[y][::-1]) for y in range(BOARD_SIZE-1, -1, -1)),
            tuple(tuple(board[x][y] for x in range(BOARD_SIZE)) for y in range(BOARD_SIZE)),
            tuple(tuple(board[x][y] for x in range(BOARD_SIZE-1, -1, -1)) for y in range(BOARD_SIZE)),
            tuple(tuple(board[x][y] for x in range(BOARD_SIZE)) for y in range(BOARD_SIZE-1, -1, -1)),
            tuple(tuple(board[x][y] for x in range(BOARD_SIZE-1, -1, -1)) for y in range(BOARD_SIZE-1, -1, -1)),
        ]
        min_hash = min(boards)
        return (min_hash, player)

    @staticmethod
    def _alpha_beta_recursive(current_board, current_player, current_depth, current_alpha, current_beta,
                              transposition_table, killer_moves, history_heuristic, pvs_flag, weights_config):
        """Recursive Alpha-Beta search function, with local (non-shared) tables."""
        key = ReversiGame._board_hash(current_board, current_player)
        
        if key in transposition_table:
            entry = transposition_table[key]
            if entry['depth'] >= current_depth:
                return entry['value']

        moves = ReversiGame._get_valid_moves(current_board, current_player)
        
        if current_depth == 0 or not moves:
            # Pass a reasonable quiescence depth limit, e.g., 3
            return ReversiGame._quiescence(current_board, current_player, current_alpha, current_beta, ReversiGame._evaluate_board, weights_config, q_depth_limit=3)

        if current_depth >= 2 and len(moves) > 0:
            null_board = [row[:] for row in current_board]
            null_score = -ReversiGame._alpha_beta_recursive(null_board, 3 - current_player, current_depth - 1 - (current_depth // 3),
                                              -current_beta, -current_beta + 1,
                                              transposition_table, killer_moves, history_heuristic, False, weights_config)
            if null_score >= current_beta:
                return null_score
                
        ordered_moves = []
        killer_move = killer_moves.get(current_depth)
        if killer_move and killer_move in moves:
            ordered_moves.append(killer_move)
            moves.remove(killer_move)
        
        moves_with_history = [(history_heuristic.get((key, m), 0), m) for m in moves]
        moves_with_history.sort(key=lambda x: x[0], reverse=True)
        ordered_moves.extend([m for _, m in moves_with_history])

        if current_player == 1: # Maximizing player (Black)
            value = -float('inf')
            for i, (x, y) in enumerate(ordered_moves):
                new_board = [row[:] for row in current_board]
                ReversiGame._simulate_move(new_board, current_player, x, y)
                
                if i == 0:
                    score = -ReversiGame._alpha_beta_recursive(new_board, 3 - current_player, current_depth - 1,
                                                  -current_beta, -current_alpha,
                                                  transposition_table, killer_moves, history_heuristic, pvs_flag, weights_config)
                else:
                    score = -ReversiGame._alpha_beta_recursive(new_board, 3 - current_player, current_depth - 1,
                                                  -current_alpha - 1, -current_alpha,
                                                  transposition_table, killer_moves, history_heuristic, False, weights_config)
                    if score > current_alpha and score < current_beta:
                        score = -ReversiGame._alpha_beta_recursive(new_board, 3 - current_player, current_depth - 1,
                                                  -current_beta, -current_alpha,
                                                  transposition_table, killer_moves, history_heuristic, pvs_flag, weights_config)
                
                value = max(value, score)
                current_alpha = max(current_alpha, value)
                
                if current_alpha >= current_beta:
                    killer_moves[current_depth] = (x, y)
                    history_heuristic[(key, (x, y))] = history_heuristic.get((key, (x, y)), 0) + current_depth
                    break
            
            transposition_table[key] = {'value': value, 'depth': current_depth}
            return value
        
        else: # Minimizing player (White)
            value = float('inf')
            for i, (x, y) in enumerate(ordered_moves):
                new_board = [row[:] for row in current_board]
                ReversiGame._simulate_move(new_board, current_player, x, y)
                
                if i == 0:
                    score = -ReversiGame._alpha_beta_recursive(new_board, 3 - current_player, current_depth - 1,
                                                  -current_beta, -current_alpha,
                                                  transposition_table, killer_moves, history_heuristic, pvs_flag, weights_config)
                else:
                    score = -ReversiGame._alpha_beta_recursive(new_board, 3 - current_player, current_depth - 1,
                                                  -current_alpha - 1, -current_alpha,
                                                  transposition_table, killer_moves, history_heuristic, False, weights_config)
                    if score > current_alpha and score < current_beta:
                        score = -ReversiGame._alpha_beta_recursive(new_board, 3 - current_player, current_depth - 1,
                                                  -current_beta, -current_alpha,
                                                  transposition_table, killer_moves, history_heuristic, pvs_flag, weights_config)
                                                
                value = min(value, score)
                current_beta = min(current_beta, value)
                
                if current_alpha >= current_beta:
                    killer_moves[current_depth] = (x, y)
                    history_heuristic[(key, (x, y))] = history_heuristic.get((key, (x, y)), 0) + current_depth
                    break
            
            transposition_table[key] = {'value': value, 'depth': current_depth}
            return value

    @staticmethod
    def _alpha_beta_worker(board_copy, player, depth, alpha, beta, pvs_flag, top_level_move, weights_config):
        """
        Worker function for multiprocessing pool to run AlphaBeta search for a single top-level move.
        Each worker has its own local transposition_table, killer_moves, history_heuristic.
        """
        # Initialize local search tables for this process
        local_transposition_table = {}
        local_killer_moves = {}
        local_history_heuristic = {}

        # Simulate the top-level move on the board copy
        ReversiGame._simulate_move(board_copy, player, top_level_move[0], top_level_move[1])

        # Start the recursive Alpha-Beta search for the resulting board state (from opponent's perspective)
        score = -ReversiGame._alpha_beta_recursive(board_copy, 3 - player, depth - 1,
                                      -beta, -alpha,
                                      local_transposition_table, local_killer_moves, local_history_heuristic, pvs_flag, weights_config)
        
        return score, top_level_move

    @staticmethod
    def _mcts_playout_static(board, player):
        current_board = [row[:] for row in board]
        current_player = player
        while True:
            moves = ReversiGame._get_valid_moves(current_board, current_player)
            if not moves:
                current_player = 3 - current_player
                moves = ReversiGame._get_valid_moves(current_board, current_player)
                if not moves:
                    black_count = sum(row.count(1) for row in current_board)
                    white_count = sum(row.count(2) for row in current_board)
                    if black_count > white_count: return 1
                    elif white_count > black_count: return 2
                    else: return 0
            move_x, move_y = random.choice(moves)
            ReversiGame._simulate_move(current_board, current_player, move_x, move_y)
            current_player = 3 - current_player

    @staticmethod
    def _self_play_game_worker(game_id, initial_board, initial_player, ai_depth, ai_mode, current_weights_copy):
        """自己対戦ゲームを実行するワーカー関数（マルチプロセス用）"""
        game_board = [row[:] for row in initial_board]
        current_player_in_game = initial_player
        game_over_in_game = False
        
        # 各プロセスで独立した探索テーブルを持つ
        local_ab_trans_table = {}
        local_ab_killer_moves = {}
        local_ab_history_heuristic = {}
        
        while not game_over_in_game:
            moves = ReversiGame._get_valid_moves(game_board, current_player_in_game)
            if not moves:
                current_player_in_game = 3 - current_player_in_game # パス
                moves = ReversiGame._get_valid_moves(game_board, current_player_in_game)
                if not moves: # 両者パスでゲーム終了
                    game_over_in_game = True
                    break

            best_move_for_turn = None
            if ai_mode == "AlphaBeta":
                best_score = -float('inf')
                for x, y in moves:
                    temp_board = [row[:] for row in game_board]
                    ReversiGame._simulate_move(temp_board, current_player_in_game, x, y)
                    
                    score = -ReversiGame._alpha_beta_recursive(temp_board, 3 - current_player_in_game, ai_depth -1,
                                                  -float('inf'), float('inf'),
                                                  local_ab_trans_table, local_ab_killer_moves, local_ab_history_heuristic, True, current_weights_copy)
                    
                    if score > best_score:
                        best_score = score
                        best_move_for_turn = (x,y)
                    elif score == best_score and random.random() < 0.5:
                        best_move_for_turn = (x,y)

            elif ai_mode == "MCTS":
                # MCTSシミュレーション数をai_depthに連動させる
                mcts_sims_per_turn = ai_depth * 10 # ai_depthに応じてシミュレーション数を増やす
                move_scores = {move: 0 for move in moves}
                for m_x, m_y in moves:
                    for _ in range(mcts_sims_per_turn):
                        sim_board = [row[:] for row in game_board]
                        ReversiGame._simulate_move(sim_board, current_player_in_game, m_x, m_y)
                        winner = ReversiGame._mcts_playout_static(sim_board, 3 - current_player_in_game)
                        if winner == current_player_in_game:
                            move_scores[(m_x, m_y)] += 1
                if move_scores:
                    best_move_for_turn = max(move_scores, key=move_scores.get)
                else:
                    best_move_for_turn = None
            
            if best_move_for_turn:
                ReversiGame._simulate_move(game_board, current_player_in_game, best_move_for_turn[0], best_move_for_turn[1])
            current_player_in_game = 3 - current_player_in_game

        black_count = sum(row.count(1) for row in game_board)
        white_count = sum(row.count(2) for row in game_board)
        
        return black_count - white_count # 黒の視点でのスコアを返す (正なら黒勝ち、負なら白勝ち)


    def save_state(self):
        """現在のゲーム状態を保存し、Undo/Redoスタックに追加するためのタプルを返す"""
        import copy
        return (copy.deepcopy(self.board), self.current_player, self.dark_mode, self.ai_depth, copy.deepcopy(self.weights))

    def load_state(self, state):
        """保存されたゲーム状態をロードする"""
        self.board, self.current_player, self.dark_mode, self.ai_depth, self.weights = state
        self.game_over = False
        self.best_move = None # ロード時にAIの最善手をクリア
        self.hint_mode = False # ロード時にヒントモードをオフ
        self.ponder_result = None # 先読み結果もクリア
        self.ponder_move = None

    def get_valid_moves(self, board=None, player=None):
        """指定されたボードとプレイヤーの合法手を返す。省略された場合は現在のゲーム状態を使用。"""
        if board is None:
            board = self.board
        if player is None:
            player = self.current_player
        return ReversiGame._get_valid_moves(board, player)

    def is_valid_move(self, x, y):
        """指定された座標が現在のプレイヤーにとって合法手であるかを判定する。"""
        return ReversiGame._is_valid_move_for(self.board, self.current_player, x, y)

    def animate_flips(self, flips, player):
        """石がひっくり返るアニメーションを表示する"""
        # 初めにボード全体を描画して背景とグリッドを確保
        self.draw_board() # これにより背景、グリッド、初期の石/ヒントが描画される
        pygame.display.flip() # アニメーション開始前にボード全体が表示されることを保証

        for fx, fy in flips:
            # アニメーションの各ステップでボードの状態を更新
            self.board[fy][fx] = player
            self.draw_ui() # UIを再描画
            self.draw_board() # ボード全体を再描画（ちらつきが発生する可能性がありますが、確実な更新のため）
            pygame.display.flip() # 画面を更新

            for _ in range(5):  # 約50ms待機しつつイベント処理
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                time.sleep(0.01)

    def make_move(self, x, y, animate=True):
        """指定された座標に手を打ち、盤面を更新する。成功した場合はTrueを返す。"""
        if not self.is_valid_move(x, y):
            return False
        
        # 状態を保存してUndoスタックに追加
        self.undo_stack.append(self.save_state())
        self.redo_stack.clear() # 新しい手なのでRedoスタックはクリア

        # 石を置く
        self.board[y][x] = self.current_player
        
        # ひっくり返す石を特定し、ひっくり返す
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        all_flips = []
        opponent = 3 - self.current_player
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            stones = [] # <-- ここでリストを初期化
            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if self.board[ny][nx] == opponent:
                    stones.append((nx, ny))
                elif self.board[ny][nx] == self.current_player:
                    # 挟めた場合、リスト内の石をひっくり返す
                    for fx, fy in stones:
                        self.board[fy][fx] = self.current_player
                        all_flips.append((fx, fy))
                    break # この方向の探索を終了
                else:
                    break # 空のマスまたは自分の石で挟めない場合
                nx += dx
                ny += dy
        
        # アニメーション表示
        if animate and all_flips:
            self.animate_flips(all_flips, self.current_player)
        
        # 効果音再生
        if self.flip_sound:
            try:
                self.flip_sound.play()
            except Exception as e:
                print(f"[Sound Play Error] 効果音の再生に失敗しました: {e}")

        # プレイヤーを切り替える
        self.current_player = 3 - self.current_player
        
        # パス判定
        if not self.has_valid_moves():
            print(f"プレイヤー{'黒' if self.current_player == 1 else '白'}に合法手がありません。パスします。")
            self.current_player = 3 - self.current_player # もう一度プレイヤーを切り替える
            if not self.has_valid_moves(): # 両者パスでゲーム終了
                self.game_over = True
                print("ゲーム終了: 両者パス。")
        
        self.best_move = None # 手を打ったらAIの最善手をクリア
        self.hint_mode = False # 手を打ったらヒントモードをオフ
        self.ponder_result = None # 手を打ったら先読み結果をクリア
        self.ponder_move = None
        return True

    def undo(self):
        """一手前の状態に戻す"""
        if self.undo_stack:
            self.redo_stack.append(self.save_state()) # 現在の状態をRedoスタックに保存
            self.load_state(self.undo_stack.pop()) # Undoスタックから前の状態をロード
            self.stop_pondering() # Undoしたら先読みを停止
            print("Undoしました。")
        else:
            print("これ以上Undoできません。")

    def redo(self):
        """一手進める"""
        if self.redo_stack:
            self.undo_stack.append(self.save_state()) # 現在の状態をUndoスタックに保存
            self.load_state(self.redo_stack.pop()) # Redoスタックから次の状態をロード
            self.stop_pondering() # Redoしたら先読みを停止
            print("これ以上Redoできません。")
        else:
            print("これ以上Redoできません。")

    def has_valid_moves(self):
        """現在のプレイヤーに合法手があるかを判定する"""
        return bool(self.get_valid_moves(self.board, self.current_player))

    def reset_board(self):
        """ボードを初期状態にリセットする"""
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.board[3][3] = self.board[4][4] = 2
        self.board[3][4] = self.board[4][3] = 1
        self.current_player = 1
        self.game_over = False
        self.undo_stack = []
        self.redo_stack = []
        self.best_move = None
        self.hint_mode = False
        self.ponder_result = None
        self.ponder_move = None
        # 重みも初期値に戻す
        self.weights = [
            [150, -40, 30, 10, 10, 30, -40, 150],
            [-40, -60, -5, -5, -5, -5, -60, -40],
            [30, -5, 20, 3, 3, 20, -5, 30],
            [10, -5, 3, 3, 3, 3, -5, 10],
            [10, -5, 3, 3, 3, 3, -5, 10],
            [30, -5, 20, 3, 3, 20, -5, 30],
            [-40, -60, -5, -5, -5, -5, -60, -40],
            [150, -40, 30, 10, 10, 30, -40, 150],
        ]
        self.stop_pondering() # リセットしたら先読みを停止
        print("ゲームをリセットしました。")

    def draw_ui(self):
        """UI要素を描画する"""
        bg_color = DARK_BG if self.dark_mode else LIGHT_BG
        self.screen.fill(bg_color)

        # タイトル "PyReversi"
        title_surface = self.large_font.render("PyReversi", True, BLACK if not self.dark_mode else WHITE)
        self.screen.blit(title_surface, (20, 10))

        # バージョン表示
        version_surface = self.small_font.render("v0.14", True, GRAY if not self.dark_mode else LIGHT_GRAY)
        self.screen.blit(version_surface, (WINDOW_WIDTH - version_surface.get_width() - 20, 18))

        # スコア表示
        black_score = sum(row.count(1) for row in self.board)
        white_score = sum(row.count(2) for row in self.board)
        
        black_score_text = f"Black: {black_score}"
        white_score_text = f"White: {white_score}"

        black_score_surface = self.font.render(black_score_text, True, BLACK if not self.dark_mode else WHITE)
        white_score_surface = self.font.render(white_score_text, True, BLACK if not self.dark_mode else WHITE)
        
        self.screen.blit(black_score_surface, (20, 50))
        self.screen.blit(white_score_surface, (WINDOW_WIDTH - white_score_surface.get_width() - 20, 50))

        # スコアの丸
        pygame.draw.circle(self.screen, BLACK, (black_score_surface.get_width() + 30, 65), 8) # Black
        pygame.draw.circle(self.screen, WHITE, (WINDOW_WIDTH - white_score_surface.get_width() - 30, 65), 8) # White
        pygame.draw.circle(self.screen, BLACK, (WINDOW_WIDTH - white_score_surface.get_width() - 30, 65), 8, 1) # White outline

        # ターン表示
        turn_player_name = "Black" if self.current_player == 1 else "White"
        turn_text = f"Turn: {turn_player_name}"
        turn_text_surface = self.large_font.render(turn_text, True, WHITE) # テキストは常に白
        
        turn_bg_color = BLACK if self.current_player == 1 else (100, 100, 100) # 黒番は黒、白番は濃いグレー
        
        turn_rect = turn_text_surface.get_rect(center=(WINDOW_WIDTH // 2, 65))
        turn_rect = turn_rect.inflate(20, 10) # パディングを追加
        pygame.draw.rect(self.screen, turn_bg_color, turn_rect, border_radius=5)
        self.screen.blit(turn_text_surface, turn_rect.move(10, 5)) # パディングを考慮して配置

        # ボードの座標ラベル (ボード描画時に描画されるため、ここでは不要)
        
        # Bottom UI: Buttons and Player Settings (下部UI: ボタンとプレイヤー設定)
        button_y_start = TOP_UI + BOARD_PIXEL + 20
        button_width, button_height = 90, 36
        button_spacing = 10
        
        # ボタンの配置
        btns_labels = ["Reset", "Undo", "Redo", "Hint"]
        total_btns_width = (button_width + button_spacing) * len(btns_labels) - button_spacing
        start_x = (WINDOW_WIDTH - total_btns_width) // 2

        self.button_rects = {} # Clear for redraw
        for i, label in enumerate(btns_labels):
            rect = pygame.Rect(start_x + i * (button_width + button_spacing), button_y_start, button_width, button_height)
            self.button_rects[label] = rect
            # "Hint"ボタンは特殊な色にする
            is_active = (label == "Hint" and self.hint_mode)
            self.draw_button(rect, label, is_active=is_active)

        # Player Type Selectors (プレイヤータイプ選択)
        player_settings_y = button_y_start + button_height + 20
        
        # Player 1 (Black)
        player1_label_x = 40
        player1_label_y = player_settings_y
        # プレイヤー表示をキャラクター文字に変更 (ダークモードに応じて切り替え)
        if not self.dark_mode:
            player1_text_surface = self.small_font.render("Player 1 (Black⚫️):", True, BLACK)
        else:
            player1_text_surface = self.small_font.render("Player 1 (Black⚫️):", True, WHITE)
        self.screen.blit(player1_text_surface, (player1_label_x, player1_label_y))
        
        radio_btn_x_offset = player1_label_x + player1_text_surface.get_width() + 10
        self.draw_radio_button(radio_btn_x_offset, player1_label_y, "Human", 1, "Human")
        self.draw_radio_button(radio_btn_x_offset + 80, player1_label_y, "AI", 1, "AI")

        # Player 2 (White)
        player2_label_x = 40
        player2_label_y = player_settings_y + 40
        # プレイヤー表示をキャラクター文字に変更 (ダークモードに応じて切り替え)
        if not self.dark_mode:
            player2_text_surface = self.small_font.render("Player 2 (White⚪️):", True, BLACK)
        else:
            player2_text_surface = self.small_font.render("Player 2 (WHite⚪️):", True, WHITE)
        self.screen.blit(player2_text_surface, (player2_label_x, player2_label_y))

        radio_btn_x_offset = player2_label_x + player2_text_surface.get_width() + 10
        self.draw_radio_button(radio_btn_x_offset, player2_label_y, "Human", 2, "Human")
        self.draw_radio_button(radio_btn_x_offset + 80, player2_label_y, "AI", 2, "AI")

        # AI Depth Slider (AI深度スライダー)
        slider_label_x = 40
        slider_label_y = player_settings_y + 100
        depth_label_surface = self.small_font.render(f"AI Depth: {self.ai_depth}", True, BLACK if not self.dark_mode else WHITE)
        self.screen.blit(depth_label_surface, (slider_label_x, slider_label_y))

        slider_track_x = slider_label_x + depth_label_surface.get_width() + 10
        slider_track_width = 360 # スライダーの長さを2倍に
        slider_track_height = 6
        slider_track_rect = pygame.Rect(slider_track_x, slider_label_y + depth_label_surface.get_height() // 2 - slider_track_height // 2, slider_track_width, slider_track_height)
        pygame.draw.rect(self.screen, GRAY, slider_track_rect, border_radius=3)

        knob_radius = 10
        # ai_depthは1からMAX_AI_DEPTHまで
        knob_x_pos = slider_track_rect.x + (self.ai_depth - 1) * (slider_track_width / (MAX_AI_DEPTH - 1))
        self.slider_rect = pygame.Rect(knob_x_pos - knob_radius, slider_track_rect.centery - knob_radius, knob_radius * 2, knob_radius * 2)
        pygame.draw.circle(self.screen, BLUE, self.slider_rect.center, knob_radius)
        
        # Toggle Dark Mode button (ダークモード切り替えボタン)
        # サイズを1.2倍に拡大
        dark_mode_btn_width = int(160 * 1.2)
        dark_mode_btn_height = int(40 * 1.2)
        dark_mode_btn_x = 40 # 左詰めに変更
        dark_mode_btn_y = slider_label_y + 60
        dark_mode_rect = pygame.Rect(dark_mode_btn_x, dark_mode_btn_y, dark_mode_btn_width, dark_mode_btn_height)
        self.button_rects["Toggle Dark Mode"] = dark_mode_rect
        self.draw_button(dark_mode_rect, "Toggle Dark Mode", is_active=self.dark_mode)

        # AI Mode buttons (AIモード切り替えボタン)
        ai_mode_btn_width = 100
        ai_mode_btn_height = 36
        ai_mode_spacing = 10
        # 右詰めに配置
        mcts_mode_rect = pygame.Rect(WINDOW_WIDTH - 40 - ai_mode_btn_width, dark_mode_btn_y, ai_mode_btn_width, ai_mode_btn_height)
        ab_mode_rect = pygame.Rect(mcts_mode_rect.x - ai_mode_btn_width - ai_mode_spacing, dark_mode_btn_y, ai_mode_btn_width, ai_mode_btn_height)
        
        self.button_rects["AlphaBeta"] = ab_mode_rect
        self.button_rects["MCTS"] = mcts_mode_rect

        self.draw_button(ab_mode_rect, "AlphaBeta", is_active=(self.ai_mode == "AlphaBeta"))
        self.draw_button(mcts_mode_rect, "MCTS", is_active=(self.ai_mode == "MCTS"))

        # 勝敗判定表示 (ゲーム終了時のみ)
        if self.game_over:
            black_final_score = sum(row.count(1) for row in self.board)
            white_final_score = sum(row.count(2) for row in self.board)
            
            result_text = ""
            result_color = BLACK if not self.dark_mode else WHITE

            if black_final_score > white_final_score:
                result_text = "Black Wins!"
                result_color = BLACK # 黒勝ち
            elif white_final_score > black_final_score:
                result_text = "White Wins!"
                result_color = WHITE # 白勝ち
            else:
                result_text = "Draw!"
                result_color = GRAY # 引き分け

            result_surface = self.large_font.render(result_text, True, result_color)
            result_rect = result_surface.get_rect(center=(WINDOW_WIDTH // 2, TOP_UI + BOARD_PIXEL + 150)) # ボードの下中央に表示
            self.screen.blit(result_surface, result_rect)


    def draw_button(self, rect, text, is_active=False, is_disabled=False):
        """汎用ボタンを描画する"""
        button_color = (100, 160, 255) if not self.dark_mode else (80, 120, 200) # デフォルト色
        text_color = WHITE # テキストは常に白

        if is_disabled:
            button_color = GRAY
            text_color = (100, 100, 100)
        elif is_active:
            button_color = YELLOW if not self.dark_mode else BLUE # アクティブ時の色
            text_color = BLACK if not self.dark_mode else WHITE # アクティブ時のテキスト色
        
        pygame.draw.rect(self.screen, button_color, rect, border_radius=8)
        text_surface = self.font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw_radio_button(self, x, y, text, player_num, value):
        """ラジオボタンを描画する"""
        radio_size = 15
        radio_padding = 5
        
        radio_rect = pygame.Rect(x, y + (self.small_font.get_height() - radio_size) // 2, radio_size, radio_size)
        pygame.draw.circle(self.screen, BLACK if not self.dark_mode else WHITE, radio_rect.center, radio_size // 2, 2)
        
        # 修正箇所: player_typesを辞書としてアクセス
        is_selected = (self.player_types[player_num] == value)
        if is_selected:
            pygame.draw.circle(self.screen, BLACK if not self.dark_mode else WHITE, radio_rect.center, radio_size // 2 - 4)
        
        text_surface = self.small_font.render(text, True, BLACK if not self.dark_mode else WHITE)
        self.screen.blit(text_surface, (x + radio_size + radio_padding, y))
        
        # player_btn_rectsにラジオボタンのRectと情報を保存
        # 0: Human, 1: AI
        btn_index = 0 if value == "Human" else 1
        self.player_btn_rects[player_num - 1][btn_index] = radio_rect


    def draw_board(self):
        """ボードと石、合法手、ヒントを描画する"""
        # ボードの背景色
        board_bg_color = GREEN if not self.dark_mode else DARK_GREEN_BOARD
        line_color = BLACK if not self.dark_mode else WHITE
        
        pygame.draw.rect(self.screen, board_bg_color, (40, TOP_UI, BOARD_PIXEL, BOARD_PIXEL))

        # グリッド線
        for i in range(BOARD_SIZE + 1):
            pygame.draw.line(self.screen, line_color, (40 + i * CELL_SIZE, TOP_UI), (40 + i * CELL_SIZE, TOP_UI + BOARD_PIXEL), 2)
            pygame.draw.line(self.screen, line_color, (40, TOP_UI + i * CELL_SIZE), (40 + BOARD_PIXEL, TOP_UI + i * CELL_SIZE), 2)

        # 座標ラベル (A-H, 1-8)
        for i in range(BOARD_SIZE):
            col_label = self.small_font.render(chr(ord('A') + i), True, line_color)
            self.screen.blit(col_label, (40 + i * CELL_SIZE + CELL_SIZE // 2 - col_label.get_width() // 2, TOP_UI - 25))
            row_label = self.small_font.render(str(i + 1), True, line_color)
            self.screen.blit(row_label, (15, TOP_UI + i * CELL_SIZE + CELL_SIZE // 2 - row_label.get_height() // 2))


        # 石の描画
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                center_x = 40 + x * CELL_SIZE + CELL_SIZE // 2
                center_y = TOP_UI + y * CELL_SIZE + CELL_SIZE // 2
                if self.board[y][x] == 1: # Black
                    pygame.draw.circle(self.screen, BLACK, (center_x, center_y), CELL_SIZE // 2 - 5)
                elif self.board[y][x] == 2: # White
                    pygame.draw.circle(self.screen, WHITE, (center_x, center_y), CELL_SIZE // 2 - 5)
                    pygame.draw.circle(self.screen, BLACK, (center_x, center_y), CELL_SIZE // 2 - 5, 1) # Outline

        # 合法手の常時表示 (Humanプレイヤーの時のみ)
        if not self.game_over and self.player_types[self.current_player] == "Human":
            valid_moves = self.get_valid_moves()
            for x, y in valid_moves:
                center_x = 40 + x * CELL_SIZE + CELL_SIZE // 2
                center_y = TOP_UI + y * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(self.screen, YELLOW, (center_x, center_y), CELL_SIZE // 4, 3) # 黄色い円で表示
        
        # Hint (最善手) の表示
        if self.hint_mode and self.best_move and self.player_types[self.current_player] == "Human":
            x, y = self.best_move
            center_x = 40 + x * CELL_SIZE + CELL_SIZE // 2
            center_y = TOP_UI + y * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, BLUE, (center_x, center_y), CELL_SIZE // 3, 4) # 青い円で最善手

        pygame.display.update() # 画面全体を更新

    def find_best_move(self, player, max_time=2.0):
        """Alpha-Beta Pruningを使って最善手を見つける"""
        moves = self.get_valid_moves(self.board, player)
        if not moves:
            return None

        empty_count = sum(row.count(0) for row in self.board)
        # 残り石数が少ない場合は、AI深度を自動調整
        if empty_count <= 15: # 残り15マス以下で完全読みに切り替え
            max_depth = empty_count
        else:
            max_depth = self.ai_depth

        tasks = []
        for x, y in moves:
            # 各タスクにAlphaBetaの引数を渡す
            tasks.append(([row[:] for row in self.board], player, max_depth, -float('inf'), float('inf'), True, (x, y), self.weights))

        if not tasks:
            return None

        results = []
        try:
            # マルチプロセスでAlphaBetaワーカーを実行
            results = self.process_pool.starmap(ReversiGame._alpha_beta_worker, tasks)
        except Exception as e:
            print(f"AlphaBeta並列処理でエラーが発生しました。シリアル処理にフォールバックします: {e}")
            # エラー時はシリアル処理にフォールバック
            for args in tasks:
                results.append(ReversiGame._alpha_beta_worker(*args)) # *args でタプルを展開して渡す

        best_score = -float('inf')
        best_move = None

        for score, move in results:
            if score > best_score:
                best_score = score
                best_move = move
            elif score == best_score:
                # 同点の場合、ランダムに選択して多様性を持たせる
                if random.random() < 0.5:
                    best_move = move
        return best_move

    def mcts_best_move(self, player, simulations_multiplier=200): # AI Depthに連動させるための引数を追加
        """Monte Carlo Tree Search (MCTS) を使って最善手を見つける"""
        import math

        moves = self.get_valid_moves(self.board, player)
        if not moves:
            return None
        
        # AI Depthに比例してシミュレーション回数を決定
        simulations = self.ai_depth * simulations_multiplier
        if simulations == 0: # 深度1の場合でも最低限のシミュレーションを保証
            simulations = simulations_multiplier 

        def run_simulation(move):
            win_count = 0
            # 各合法手に対して均等にシミュレーションを割り振る
            sim_per_move = math.ceil(simulations / len(moves)) 
            for _ in range(sim_per_move):
                board_copy = [row[:] for row in self.board]
                ReversiGame._simulate_move(board_copy, player, move[0], move[1])
                winner = ReversiGame._mcts_playout_static(board_copy, 3 - player) # 相手のターンからプレイアウト開始
                if winner == player: # 自分のプレイヤーが勝った場合
                    win_count += 1
            return (move, win_count)

        results = []
        # スレッド数の決定（moves数に応じて適切な数に）
        n_workers = min(self.max_workers, len(moves))
        if n_workers <= 1 or not moves: # スレッド数が1以下、または合法手がない場合はシリアル実行
            results = [run_simulation(move) for move in moves]
        else: # 並列実行
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(run_simulation, moves))

        if not results:
            return None

        # 最も勝利回数の多い手を選択
        best_move, best_wins = max(results, key=lambda x: x[1])
        return best_move

    def mcts_playout(self, board, player):
        """MCTSのプレイアウト（ランダムなゲームシミュレーション）"""
        current_board = [row[:] for row in board] # ボードのコピー
        current_player = player # 現在のプレイヤー
        
        while True:
            moves = ReversiGame._get_valid_moves(current_board, current_player)
            if not moves:
                current_player = 3 - current_player # パス
                moves = ReversiGame._get_valid_moves(current_board, current_player)
                if not moves: # 両者パスでゲーム終了
                    black_count = sum(row.count(1) for row in current_board)
                    white_count = sum(row.count(2) for row in current_board)
                    if black_count > white_count:
                        return 1 # Black wins
                    elif white_count > black_count:
                        return 2 # White wins
                    else:
                        return 0 # Draw
            
            move_x, move_y = random.choice(moves) # ランダムに手を選択
            ReversiGame._simulate_move(current_board, current_player, move_x, move_y) # 手を打つ
            current_player = 3 - current_player # プレイヤーを切り替え

    def start_pondering(self):
        """AIがHumanの次の手を予測して先読みを開始する"""
        # 既に先読みスレッドが動いている場合は停止
        if self.ponder_thread and self.ponder_thread.is_alive():
            self.ponder_stop.set()
            self.ponder_thread.join()
        
        self.ponder_stop.clear() # 停止シグナルをクリア
        self.ponder_result = None # 先読み結果をリセット
        self.ponder_move = None # 先読みの基準となるHumanの最善手
        
        # 現在のボード状態とプレイヤーをコピー
        ponder_board = [row[:] for row in self.board]
        ponder_player = self.current_player
        ponder_depth = self.ai_depth
        ponder_ai_mode = self.ai_mode
        ponder_weights = [row[:] for row in self.weights]

        def _ponder_func():
            """先読みを行う内部関数（別スレッドで実行）"""
            print(f"先読み開始: プレイヤー {ponder_player}, 深度 {ponder_depth}, モード {ponder_ai_mode}")
            
            # Humanが打つ可能性のある手を全て列挙
            human_valid_moves = ReversiGame._get_valid_moves(ponder_board, ponder_player)
            if not human_valid_moves:
                self.ponder_result = None
                print("Humanに合法手がないため先読みをスキップ。")
                return

            # Humanが打つであろう最善手（ここでは最も評価値が高い手）を予測
            predicted_human_move = None
            if ponder_ai_mode == "AlphaBeta":
                # AlphaBetaでHumanの最善手を予測（HumanもAIと同じ評価関数を使うと仮定）
                best_human_score = -float('inf')
                for x, y in human_valid_moves:
                    temp_board = [row[:] for row in ponder_board]
                    ReversiGame._simulate_move(temp_board, ponder_player, x, y)
                    # Humanの視点での評価
                    score = ReversiGame._evaluate_board(temp_board, ponder_player, ponder_weights)
                    if score > best_human_score:
                        best_human_score = score
                        predicted_human_move = (x, y)
            else: # MCTSの場合、ここでは簡易的に最初の合法手を予測
                if human_valid_moves:
                    predicted_human_move = human_valid_moves[0] # またはランダム

            if not predicted_human_move:
                self.ponder_result = None
                print("Humanの予測手が見つからなかったため先読みをスキップ。")
                return

            self.ponder_move = predicted_human_move # 予測したHumanの手を保存

            # Humanが予測手 (predicted_human_move) を打った後の局面をシミュレート
            simulated_board_after_human_move = [row[:] for row in ponder_board]
            ReversiGame._simulate_move(simulated_board_after_human_move, ponder_player, predicted_human_move[0], predicted_human_move[1])
            
            # その局面でAIが打つべき最善手を計算
            ai_player_after_human_move = 3 - ponder_player
            
            calculated_ai_best_move = None
            if ponder_ai_mode == "AlphaBeta":
                # AlphaBetaでAIの最善手を計算
                local_transposition_table = {} # 各スレッド/プロセスで独立したテーブル
                local_killer_moves = {}
                local_history_heuristic = {}
                
                ai_moves = ReversiGame._get_valid_moves(simulated_board_after_human_move, ai_player_after_human_move)
                if not ai_moves:
                    calculated_ai_best_move = None
                else:
                    best_ai_score = -float('inf')
                    for x, y in ai_moves:
                        if self.ponder_stop.is_set(): # 停止シグナルが設定されたら中断
                            self.ponder_result = None
                            print("先読み中断: 停止シグナルを受信。")
                            return
                            
                        temp_board = [row[:] for row in simulated_board_after_human_move]
                        ReversiGame._simulate_move(temp_board, ai_player_after_human_move, x, y)
                        
                        score = -ReversiGame._alpha_beta_recursive(temp_board, 3 - ai_player_after_human_move, ponder_depth - 1,
                                                      -float('inf'), float('inf'),
                                                      local_transposition_table, local_killer_moves, local_history_heuristic, True, ponder_weights)
                        
                        if score > best_ai_score:
                            best_ai_score = score
                            calculated_ai_best_move = (x, y)
                        elif score == best_ai_score and random.random() < 0.5: # 同点ならランダム
                            calculated_ai_best_move = (x, y)

            elif ponder_ai_mode == "MCTS":
                # MCTSでAIの最善手を計算
                # AI Depthに比例してシミュレーション回数を決定
                mcts_simulations = ponder_depth * 500 # 先読み時のMCTSシミュレーション数を調整
                calculated_ai_best_move = self.mcts_best_move(ai_player_after_human_move, simulations_multiplier=mcts_simulations // ponder_depth) # multiplierを調整して渡す

            self.ponder_result = calculated_ai_best_move
            print(f"先読み完了: 予測Human手 {predicted_human_move}, AIの最善手 {calculated_ai_best_move}")

        self.ponder_thread = threading.Thread(target=_ponder_func)
        self.ponder_thread.daemon = True # メインプログラム終了時にスレッドも終了させる
        self.ponder_thread.start()

    def stop_pondering(self):
        """先読みスレッドを停止する"""
        if self.ponder_thread and self.ponder_thread.is_alive():
            self.ponder_stop.set() # 停止シグナルを設定
            self.ponder_thread.join(timeout=0.5) # スレッドが終了するまで最大0.5秒待機
            if self.ponder_thread.is_alive():
                print("警告: 先読みスレッドが時間内に終了しませんでした。")
        self.ponder_result = None
        self.ponder_move = None
        self.ponder_stop.clear() # 次の先読みのためにシグナルをクリア

    def handle_event(self, event):
        """Pygameイベントを処理する"""
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1: # 左クリック
                mx, my = event.pos
                
                # ボードクリック判定
                board_left = 40
                board_top = TOP_UI
                if (board_left <= mx < board_left + BOARD_PIXEL and board_top <= my < board_top + BOARD_PIXEL and not self.game_over):
                    x = (mx - board_left) // CELL_SIZE
                    y = (my - board_top) // CELL_SIZE
                    if self.player_types[self.current_player] == "Human":
                        if self.make_move(x, y):
                            self.stop_pondering() # Humanが手を打ったら先読みを停止
                            # AIのターンになったら、AIが手を打つ準備をする
                            # ただし、即座に打つのではなく、メインループでAIのターン処理が行われるのを待つ
                            if not self.game_over and self.player_types[self.current_player] == "AI":
                                self.start_pondering()
                                
                # UI要素クリック判定
                for label, rect in self.button_rects.items():
                    if rect.collidepoint(mx, my):
                        if label == "Reset":
                            self.reset_board()
                        elif label == "Undo":
                            self.undo()
                            # Undo/Redo後はAIのターンでも即座に打たず、メインループで処理を待つ
                            if not self.game_over and self.player_types[self.current_player] == "AI":
                                self.start_pondering()
                        elif label == "Redo":
                            self.redo()
                            # Undo/Redo後はAIのターンでも即座に打たず、メインループで処理を待つ
                            if not self.game_over and self.player_types[self.current_player] == "AI":
                                self.start_pondering()
                        elif label == "Hint":
                            self.hint_mode = not self.hint_mode # Hintモードをトグル
                            if self.hint_mode and self.player_types[self.current_player] == "Human":
                                # HintモードオンでHumanプレイヤーなら最善手を計算
                                self.best_move = self.find_best_move(self.current_player)
                                if not self.best_move:
                                    print("ヒント: 合法手がありません。")
                            else:
                                self.best_move = None # Hintモードオフなら最善手もクリア
                        elif label == "Toggle Dark Mode":
                            self.dark_mode = not self.dark_mode
                        elif label in ["AlphaBeta", "MCTS"]: # AIモード切り替えボタンの処理
                            self.ai_mode = label
                            self.stop_pondering() # AIモード変更時は先読み停止
                            if not self.game_over and self.player_types[self.current_player] == "AI":
                                self.start_pondering() # 新しいモードで先読み開始
                        # ボタンクリック後のUI更新を確実に
                        self.draw_ui()
                        self.draw_board()
                        pygame.display.flip()
                        break # 複数のボタンが重なるのを防ぐ

                # プレイヤータイプラジオボタンクリック判定
                for i in range(2): # i=0 for Player 1, i=1 for Player 2
                    for j in range(2): # j=0 for Human, j=1 for AI
                        rect = self.player_btn_rects[i][j]
                        if rect and rect.collidepoint(mx, my):
                            self.player_types[i+1] = "Human" if j == 0 else "AI"
                            self.stop_pondering() # プレイヤータイプ変更時は先読み停止
                            # AIのターンになったら先読みを開始
                            if not self.game_over and self.player_types[self.current_player] == "AI":
                                self.start_pondering()
                            self.draw_ui()
                            self.draw_board()
                            pygame.display.flip()
                            break
                    else: # 内側のループがbreakしなかった場合
                        continue
                    break # 外側のループもbreak

                # スライダーのドラッグ開始
                if self.slider_rect and self.slider_rect.collidepoint(mx, my):
                    self.slider_drag = True
        
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                self.slider_drag = False
        
        elif event.type == MOUSEMOTION:
            if self.slider_drag:
                mx, my = event.pos
                slider_x = 40 # ボードの左端に合わせる
                slider_width = BOARD_PIXEL # ボードの幅に合わせる
                
                # スライダーのハンドル位置をマウス座標に基づいて計算し、範囲内に収める
                # スライダーは1からMAX_AI_DEPTHまで
                # progressは0.0から1.0の範囲
                progress = (mx - slider_x) / slider_width
                self.ai_depth = round(1 + progress * (MAX_AI_DEPTH - 1))
                self.ai_depth = max(1, min(self.ai_depth, MAX_AI_DEPTH)) # 範囲を保証
                
                self.stop_pondering() # 深度変更時は先読み停止
                # AIのターンになったら先読みを開始
                if not self.game_over and self.player_types[self.current_player] == "AI":
                    self.start_pondering()
                self.draw_ui()
                self.draw_board()
                pygame.display.flip()


    def self_play_tuning(self, num_games=10, tuning_iterations=5):
        """AIの評価関数の重みを自己対戦でチューニングする（簡易版）"""
        print(f"自己対戦チューニング開始: {num_games}ゲーム x {tuning_iterations}イテレーション")

        for iteration in range(tuning_iterations):
            print(f"--- チューニングイテレーション {iteration + 1}/{tuning_iterations} ---")
            
            game_tasks = []
            for game_num in range(num_games):
                # 各ゲームの初期状態、AI深度、モード、現在の重みをタスクとして追加
                game_tasks.append((game_num, [row[:] for row in self.board], 1, self.ai_depth, self.ai_mode, [row[:] for row in self.weights]))
            
            # マルチプロセスプールでゲームを実行し、結果を取得
            game_results = self.process_pool.starmap(ReversiGame._self_play_game_worker, game_tasks)

            avg_score = sum(game_results) / num_games
            print(f"イテレーション {iteration + 1} 平均スコア (黒の視点): {avg_score:.2f}")

            new_weights = [row[:] for row in self.weights]
            
            # これは、洗練された重み調整アルゴリズムのプレースホルダーです。
            # デモンストレーションのため、全体的なパフォーマンスに基づいた簡単な調整を行います。
            corners = [(0,0),(0,BOARD_SIZE-1),(BOARD_SIZE-1,0),(BOARD_SIZE-1,BOARD_SIZE-1)]
            if avg_score > 5: # 黒が有利ならコーナーの重みを少し上げる
                for r, c in corners:
                    new_weights[r][c] += 1
            elif avg_score < -5: # 白が有利ならコーナーの重みを少し下げる
                for r, c in corners:
                    new_weights[r][c] -= 1
            self.weights = new_weights
            print(f"現在のAI評価重み (例: コーナー値): {self.weights[0][0]}")

        print("自己対戦チューニング完了。")

    def run(self):
        """ゲームのメインループ"""
        running = True
        
        # ゲーム開始時にAIのターンであれば先読みを開始
        if self.player_types[self.current_player] == "AI":
            self.start_pondering()

        try:
            while running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False
                    self.handle_event(event) # イベント処理はhandle_eventに集約

                if not self.game_over:
                    current_player_type = self.player_types[self.current_player]
                    if current_player_type == "AI":
                        # AIの先読み結果があればそれを利用
                        if self.ponder_result:
                            self.best_move = self.ponder_result # 先読み結果を最善手として設定
                            self.ponder_result = None # 結果をクリア
                            
                            # 最善手が有効な手であれば打つ
                            if self.best_move and self.is_valid_move(self.best_move[0], self.best_move[1]):
                                self.make_move(self.best_move[0], self.best_move[1])
                                self.stop_pondering() # 手を打ったら先読みを停止
                                # 次のプレイヤーがAIであれば再度先読みを開始
                                if not self.game_over and self.player_types[self.current_player] == "AI":
                                    self.start_pondering()
                                # UIを更新
                                self.draw_ui()
                                self.draw_board()
                            else:
                                # 先読みした手が無効な場合（稀に発生する可能性あり）
                                print(f"先読みした手 {self.best_move} が無効です。再計算します。")
                                self.best_move = None
                                self.stop_pondering() # 先読みを停止
                                # 通常のAI計算を実行
                                if self.ai_mode == "AlphaBeta":
                                    self.best_move = self.find_best_move(self.current_player)
                                elif self.ai_mode == "MCTS":
                                    self.best_move = self.mcts_best_move(self.current_player)
                                
                                if self.best_move:
                                    self.make_move(self.best_move[0], self.best_move[1])
                                    if not self.game_over and self.player_types[self.current_player] == "AI":
                                        self.start_pondering()
                                    self.draw_ui()
                                    self.draw_board()
                                else:
                                    # AIが合法手を見つけられない場合（パスやゲーム終了）
                                    pass
                        # 先読みスレッドが動いていない、または終了している場合
                        elif not self.ponder_thread or not self.ponder_thread.is_alive():
                            # AIが最善手を計算
                            if self.ai_mode == "AlphaBeta":
                                self.best_move = self.find_best_move(self.current_player)
                            elif self.ai_mode == "MCTS":
                                self.best_move = self.mcts_best_move(self.current_player)
                            
                            if self.best_move:
                                self.make_move(self.best_move[0], self.best_move[1])
                                self.stop_pondering() # 手を打ったら先読みを停止
                                # 次のプレイヤーがAIであれば再度先読みを開始
                                if not self.game_over and self.player_types[self.current_player] == "AI":
                                    self.start_pondering()
                            self.draw_ui()
                            self.draw_board()
                
                # 常にUIとボードを再描画
                self.draw_ui()
                self.draw_board()

                pygame.display.flip() # 画面全体を更新
                time.sleep(0.01) # わずかな遅延

        except Exception as e:
            print(f"メインループで予期せぬエラーが発生しました: {e}")
            import traceback
            traceback.print_exc() # スタックトレースも出力
        finally: # ゲーム終了時またはエラー発生時に実行されるクリーンアップ処理
            if self.process_pool:
                self.process_pool.terminate() # プロセスプールを終了
                self.process_pool.join() # 終了を待機
            self.stop_pondering() # 先読みスレッドを停止
            pygame.quit() # Pygameを終了
            sys.exit() # プログラムを終了

if __name__ == "__main__":
    game = ReversiGame()
    # 自己対戦チューニングを実行する場合は以下の行のコメントを解除してください:
    # game.self_play_tuning(num_games=5, tuning_iterations=3)
    game.run()