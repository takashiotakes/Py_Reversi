import pygame
import sys
import time
from pygame.locals import *
import threading
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# 色の定義
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 204, 0)
DARK_BG = (40, 40, 40)
LIGHT_BG = (230, 230, 230)
GRAY = (180, 180, 180)

# ボードのサイズ
BOARD_SIZE = 8
CELL_SIZE = 60
BOARD_PIXEL = BOARD_SIZE * CELL_SIZE

# UI領域
TOP_UI = 120
BOTTOM_UI = 260  # 以前より+80px拡大
WINDOW_WIDTH = BOARD_PIXEL + 80
WINDOW_HEIGHT = TOP_UI + BOARD_PIXEL + BOTTOM_UI

FONT_NAME = None

class ReversiGame:
    def __init__(self):
        pygame.init()
        # CPU/スレッド情報の取得
        import os
        try:
            self.cpu_count = len(os.sched_getaffinity(0))  # 利用可能なCPUコア数
        except AttributeError:
            self.cpu_count = multiprocessing.cpu_count()  # フォールバック
        self.max_workers = max(1, min(self.cpu_count - 1, 4))  # メインスレッド用に1コア残す、最大4
        print(f"利用可能なCPUコア数: {self.cpu_count}")
        print(f"AI思考用スレッド数 (MCTS): {self.max_workers}")
        self.process_pool = None
        # 既存のコードは以下へ
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
        pygame.display.set_caption('Reversi')
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.board[3][3] = self.board[4][4] = 2  # White
        self.board[3][4] = self.board[4][3] = 1  # Black
        self.current_player = 1  # Black starts
        self.game_over = False
        self.dark_mode = False
        self.ai_depth = 3
        self.hint_mode = False
        self.undo_stack = []
        self.redo_stack = []
        self.player_types = ["Human", "AI"]
        self.best_move = None
        self.button_rects = {}
        self.slider_drag = False
        self.slider_rect = None
        self.player_btn_rects = [[None, None], [None, None]]
        self.transposition_table = {}
        self.ponder_thread = None
        self.ponder_result = None
        self.ponder_move = None
        self.ponder_stop = threading.Event()
        self.killer_moves = {}
        self.history_heuristic = {}
        self.ai_mode = "AlphaBeta"  # "AlphaBeta" or "MCTS"

        # 効果音の初期化
        try:
            pygame.mixer.init()
            self.flip_sound = pygame.mixer.Sound("flip.mp3")
        except Exception as e:
            print(f"[Sound Error] {e}")
            self.flip_sound = None

    def save_state(self):
        # Undo/Redo用に盤面・プレイヤー状態を保存（player_typesは保存しない）
        import copy
        return (copy.deepcopy(self.board), self.current_player, self.dark_mode, self.ai_depth)

    def load_state(self, state):
        self.board, self.current_player, self.dark_mode, self.ai_depth = state
        self.game_over = False
        self.best_move = None

    def get_valid_moves(self, board=None, player=None):
        if board is None:
            board = self.board
        if player is None:
            player = self.current_player
        moves = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.is_valid_move_for(board, player, x, y):
                    moves.append((x, y))
        return moves

    def is_valid_move_for(self, board, player, x, y):
        if board[y][x] != 0:
            return False
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dx, dy in directions:
            if self._can_flip_for(board, player, x, y, dx, dy):
                return True
        return False

    def _can_flip_for(self, board, player, x, y, dx, dy):
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

    def animate_flips(self, flips, player):
        for fx, fy in flips:
            self.board[fy][fx] = player
            self.draw_board()
            pygame.display.update()
            for _ in range(5):  # 約50ms待機しつつイベント処理
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                time.sleep(0.01)

    def make_move(self, x, y, animate=True):
        if not self.is_valid_move(x, y):
            return False
        self.undo_stack.append(self.save_state())
        self.redo_stack.clear()
        self.board[y][x] = self.current_player
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        all_flips = []
        for dx, dy in directions:
            flips = self._get_flips(x, y, dx, dy)
            if flips:
                all_flips.extend(flips)
        if animate and all_flips:
            self.animate_flips(all_flips, self.current_player)
        else:
            for fx, fy in all_flips:
                self.board[fy][fx] = self.current_player
        # 効果音再生
        if self.flip_sound:
            try:
                self.flip_sound.play()
            except Exception as e:
                print(f"[Sound Play Error] {e}")
        self.current_player = 3 - self.current_player
        if not self.has_valid_moves():
            self.current_player = 3 - self.current_player
            if not self.has_valid_moves():
                self.game_over = True
        self.best_move = None
        return True

    def _get_flips(self, x, y, dx, dy):
        opponent = 3 - self.current_player
        flips = []
        nx, ny = x + dx, y + dy
        while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if self.board[ny][nx] == opponent:
                flips.append((nx, ny))
            elif self.board[ny][nx] == self.current_player:
                return flips if flips else []
            else:
                break
            nx += dx
            ny += dy
        return []

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.save_state())
            self.load_state(self.undo_stack.pop())

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.save_state())
            self.load_state(self.redo_stack.pop())

    def _can_flip(self, x, y, dx, dy):
        opponent = 3 - self.current_player
        x, y = x + dx, y + dy
        found_opponent = False
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            if self.board[y][x] == opponent:
                found_opponent = True
            elif self.board[y][x] == self.current_player:
                return found_opponent
            else:
                break
            x, y = x + dx, y + dy
        return False

    def _flip_stones(self, x, y, dx, dy):
        opponent = 3 - self.current_player
        stones_to_flip = []
        x, y = x + dx, y + dy
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            if self.board[y][x] == opponent:
                stones_to_flip.append((x, y))
            elif self.board[y][x] == self.current_player:
                for fx, fy in stones_to_flip:
                    self.board[fy][fx] = self.current_player
                break
            else:
                break
            x, y = x + dx, y + dy

    def has_valid_moves(self):
        return any(self.is_valid_move(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE))

    def reset_board(self):
        # 盤面と棋譜履歴のみリセット
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.board[3][3] = self.board[4][4] = 2  # 白
        self.board[3][4] = self.board[4][3] = 1  # 黒
        self.current_player = 1
        self.game_over = False
        self.undo_stack = []
        self.redo_stack = []
        self.best_move = None

    def draw_ui(self):
        bg = DARK_BG if self.dark_mode else LIGHT_BG
        self.screen.fill(bg)
        title = self.large_font.render("iOS Reversi", True, BLACK if not self.dark_mode else WHITE)
        self.screen.blit(title, (20, 10))
        ver = self.small_font.render("v0.14", True, GRAY)
        self.screen.blit(ver, (200, 18))
        # スコア（丸はテキストでなくcircleで描画）
        black_score = sum(row.count(1) for row in self.board)
        white_score = sum(row.count(2) for row in self.board)
        btxt = self.font.render(f"Black: {black_score}", True, BLACK if not self.dark_mode else WHITE)
        wtxt = self.font.render(f"White: {white_score}", True, BLACK if not self.dark_mode else WHITE)
        self.screen.blit(btxt, (20, 50))
        self.screen.blit(wtxt, (200, 50))
        # 黒石・白石の丸をcircleで描画
        pygame.draw.circle(self.screen, BLACK, (120, 65), 10)
        pygame.draw.circle(self.screen, WHITE, (320, 65), 10)
        pygame.draw.circle(self.screen, BLACK, (320, 65), 10, 1)  # 白石の枠
        # ターン表示
        turn_str = "Black" if self.current_player == 1 else "White"
        turn_fg = WHITE if not self.dark_mode else BLACK
        turn_bg = BLACK if not self.dark_mode else WHITE
        turn_mark_col = BLACK if self.current_player == 1 else WHITE
        pygame.draw.rect(self.screen, turn_bg, (WINDOW_WIDTH//2-70, 80, 140, 36), border_radius=18)
        turn_label = self.font.render(f"Turn: {turn_str}", True, turn_fg)
        self.screen.blit(turn_label, (WINDOW_WIDTH//2-60, 84))
        # ターンの丸
        pygame.draw.circle(self.screen, turn_mark_col, (WINDOW_WIDTH//2+60, 98), 10)
        if self.current_player == 2:
            pygame.draw.circle(self.screen, BLACK, (WINDOW_WIDTH//2+60, 98), 10, 1)

    def draw_board(self):
        self.draw_ui()
        board_left = 40
        board_top = TOP_UI
        board_bg = (60, 60, 60) if self.dark_mode else GREEN
        pygame.draw.rect(self.screen, board_bg, (board_left, board_top, BOARD_PIXEL, BOARD_PIXEL))
        for x in range(BOARD_SIZE+1):
            pygame.draw.line(self.screen, BLACK, (board_left + x*CELL_SIZE, board_top), (board_left + x*CELL_SIZE, board_top+BOARD_PIXEL), 2)
        for y in range(BOARD_SIZE+1):
            pygame.draw.line(self.screen, BLACK, (board_left, board_top + y*CELL_SIZE), (board_left+BOARD_PIXEL, board_top + y*CELL_SIZE), 2)
        # 盤面の石（ダークモードでも色は変えない）
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                v = self.board[y][x]
                if v > 0:
                    color = BLACK if v == 1 else WHITE
                    center = (board_left + x*CELL_SIZE + CELL_SIZE//2, board_top + y*CELL_SIZE + CELL_SIZE//2)
                    pygame.draw.circle(self.screen, color, center, CELL_SIZE//2 - 6)
        # 合法手・ヒント
        if self.player_types[self.current_player-1] == "AI":
            moves = []  # AI手番時は合法手・ヒント非表示
        else:
            moves = self.get_valid_moves()
        if self.hint_mode and self.best_move and self.player_types[self.current_player-1] != "AI":
            for x, y in moves:
                if (x, y) == self.best_move:
                    center = (40 + x*CELL_SIZE + CELL_SIZE//2, TOP_UI + y*CELL_SIZE + CELL_SIZE//2)
                    pygame.draw.circle(self.screen, (255, 140, 0), center, 13, 0)  # オレンジで最善手
                else:
                    center = (40 + x*CELL_SIZE + CELL_SIZE//2, TOP_UI + y*CELL_SIZE + CELL_SIZE//2)
                    pygame.draw.circle(self.screen, YELLOW, center, 10, 2)
        else:
            for x, y in moves:
                center = (40 + x*CELL_SIZE + CELL_SIZE//2, TOP_UI + y*CELL_SIZE + CELL_SIZE//2)
                pygame.draw.circle(self.screen, YELLOW, center, 10, 2)
        self.draw_bottom_ui()

    def draw_bottom_ui(self):
        self.button_rects = {}
        self.player_btn_rects = [[None, None], [None, None]]
        y0 = TOP_UI + BOARD_PIXEL + 20
        btn_w, btn_h = 90, 36
        btns = ["Reset", "Undo", "Redo", "Hint"]
        for i, label in enumerate(btns):
            rect = pygame.Rect(40 + i*(btn_w+10), y0, btn_w, btn_h)
            color = (100, 160, 255) if not self.dark_mode else (80, 120, 200)
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            txt = self.small_font.render(label, True, WHITE)
            self.screen.blit(txt, (rect.x+18, rect.y+7))
            self.button_rects[label] = rect
        y1 = y0 + btn_h + 18
        # Playerラベルも丸はcircleで描画
        for i, p in enumerate(["Player 1", "Player 2"]):
            txt = self.small_font.render(p, True, BLACK if not self.dark_mode else WHITE)
            self.screen.blit(txt, (40, y1 + i*32))
            cx = 140
            cy = y1 + i*32 + 14
            col = BLACK if i == 0 else WHITE
            pygame.draw.circle(self.screen, col, (cx, cy), 10)
            if i == 1:
                pygame.draw.circle(self.screen, BLACK, (cx, cy), 10, 1)
            for j, t in enumerate(["Human", "AI"]):
                rect = pygame.Rect(170 + j*70, y1 + i*32, 60, 28)
                if self.player_types[i] == t:
                    colbtn = (100, 200, 100) if not self.dark_mode else (100, 200, 200)
                else:
                    colbtn = (220,220,220) if not self.dark_mode else (80,80,80)
                pygame.draw.rect(self.screen, colbtn, rect, border_radius=6)
                ttxt = self.small_font.render(t, True, BLACK if not self.dark_mode else WHITE)
                self.screen.blit(ttxt, (rect.x+8, rect.y+4))
                self.player_btn_rects[i][j] = rect
        y2 = y1 + 2*32 + 10
        txt = self.small_font.render("AI Depth: ", True, BLACK if not self.dark_mode else WHITE)
        self.screen.blit(txt, (40, y2))
        slider_x = 140
        slider_w = 180
        self.slider_rect = pygame.Rect(slider_x, y2+10, slider_w, 20)
        pygame.draw.rect(self.screen, GRAY, (slider_x, y2+10, slider_w, 6), border_radius=3)
        knob_x = slider_x + int((self.ai_depth-1)/14*slider_w)
        pygame.draw.circle(self.screen, BLUE, (knob_x, y2+13), 10)
        dtxt = self.small_font.render(str(self.ai_depth), True, BLACK if not self.dark_mode else WHITE)
        self.screen.blit(dtxt, (slider_x+slider_w+10, y2+3))
        btn_w2, btn_h2 = 260, 48
        btn_x2 = (WINDOW_WIDTH - btn_w2) // 2
        btn_y2 = y2 + 60
        color2 = (0, 0, 0) if not self.dark_mode else (230, 230, 230)
        pygame.draw.rect(self.screen, color2, (btn_x2, btn_y2, btn_w2, btn_h2), border_radius=16)
        txt2 = self.font.render("Toggle Dark Mode", True, WHITE if not self.dark_mode else BLACK)
        self.screen.blit(txt2, (btn_x2 + 30, btn_y2 + 10))
        self.button_rects["DarkMode"] = pygame.Rect(btn_x2, btn_y2, btn_w2, btn_h2)
        # AIロジック切り替えボタン（Hintと被らず中間位置、幅縮小）
        btn_w3, btn_h3 = 90, 36  # 幅を3/4に
        # Hintボタンの右隣から少し間隔を空けて配置
        hint_rect = self.button_rects["Hint"]
        btn_x3 = hint_rect.right + 30  # Hintボタン右端+30px
        btn_y3 = y0
        for i, mode in enumerate(["AlphaBeta", "MCTS"]):
            rect = pygame.Rect(btn_x3, btn_y3 + i*(btn_h3+10), btn_w3, btn_h3)
            color = (100, 200, 100) if self.ai_mode == mode else (220,220,220)
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            txt = self.small_font.render(mode, True, BLACK if not self.dark_mode else WHITE)
            self.screen.blit(txt, (rect.x+10, rect.y+7))
            self.button_rects[mode] = rect

    def count_frontier_discs(self, board, player):
        frontier = 0
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board[y][x] == player:
                    for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                            if board[ny][nx] == 0:
                                frontier += 1
                                break
        return frontier

    def get_strong_stable_discs(self, board, player):
        # 全8方向で角から伝播する厳密な確定石判定
        stable = [[False]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        # 角から全方向に伝播
        for sx, sy in [(0,0),(0,BOARD_SIZE-1),(BOARD_SIZE-1,0),(BOARD_SIZE-1,BOARD_SIZE-1)]:
            if board[sy][sx] == player:
                stable[sy][sx] = True
        changed = True
        directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
        while changed:
            changed = False
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    if board[y][x] != player or stable[y][x]:
                        continue
                    stable_dirs = 0
                    for dx, dy in directions:
                        nx, ny = x, y
                        while 0 <= nx+dx < BOARD_SIZE and 0 <= ny+dy < BOARD_SIZE:
                            nx += dx
                            ny += dy
                            if board[ny][nx] != player:
                                break
                            if stable[ny][nx]:
                                stable_dirs += 1
                                break
                        else:
                            stable_dirs += 1
                    if stable_dirs == 8:
                        stable[y][x] = True
                        changed = True
        return sum(stable[y][x] for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)), stable

    def evaluate_board(self, board, player):
        # 進行度による重み切り替え
        empty_count = sum(row.count(0) for row in board)
        if empty_count > 44:  # 序盤
            weights = [
                [150, -40, 30, 10, 10, 30, -40, 150],
                [-40, -60, -5, -5, -5, -5, -60, -40],
                [30, -5, 20, 3, 3, 20, -5, 30],
                [10, -5, 3, 3, 3, 3, -5, 10],
                [10, -5, 3, 3, 3, 3, -5, 10],
                [30, -5, 20, 3, 3, 20, -5, 30],
                [-40, -60, -5, -5, -5, -5, -60, -40],
                [150, -40, 30, 10, 10, 30, -40, 150],
            ]
        elif empty_count > 20:  # 中盤
            weights = [
                [120, -20, 20, 5, 5, 20, -20, 120],
                [-20, -40, -5, -5, -5, -5, -40, -20],
                [20, -5, 15, 3, 3, 15, -5, 20],
                [5, -5, 3, 3, 3, 3, -5, 5],
                [5, -5, 3, 3, 3, 3, -5, 5],
                [20, -5, 15, 3, 3, 15, -5, 20],
                [-20, -40, -5, -5, -5, -5, -40, -20],
                [120, -20, 20, 5, 5, 20, -20, 120],
            ]
        else:  # 終盤
            weights = [
                [100, -10, 10, 2, 2, 10, -10, 100],
                [-10, -20, -2, -2, -2, -2, -20, -10],
                [10, -2, 8, 1, 1, 8, -2, 10],
                [2, -2, 1, 1, 1, 1, -2, 2],
                [2, -2, 1, 1, 1, 1, -2, 2],
                [10, -2, 8, 1, 1, 8, -2, 10],
                [-10, -20, -2, -2, -2, -2, -20, -10],
                [100, -10, 10, 2, 2, 10, -10, 100],
            ]
        score = 0
        opponent = 3 - player
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board[y][x] == player:
                    score += weights[y][x]
                elif board[y][x] == opponent:
                    score -= weights[y][x]
        my_count = sum(row.count(player) for row in board)
        op_count = sum(row.count(opponent) for row in board)
        score += (my_count - op_count) * 10
        my_moves = len(self.get_valid_moves(board, player))
        op_moves = len(self.get_valid_moves(board, opponent))
        score += (my_moves - op_moves) * 5
        # フロンティアディスク
        my_frontier = self.count_frontier_discs(board, player)
        op_frontier = self.count_frontier_discs(board, opponent)
        score -= (my_frontier - op_frontier) * 4
        # 厳密な確定石
        my_strong_stable, my_stable_map = self.get_strong_stable_discs(board, player)
        op_strong_stable, op_stable_map = self.get_strong_stable_discs(board, opponent)
        score += (my_strong_stable - op_strong_stable) * 30
        # フロンティアの安定性強化
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board[y][x] == player:
                    for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                            if board[ny][nx] == 0:
                                if my_stable_map[y][x]:
                                    score += 8
                                break
        # パリティ評価
        empty = [[board[y][x] == 0 for x in range(BOARD_SIZE)] for y in range(BOARD_SIZE)]
        parity_bonus = 0
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if empty[y][x]:
                    cnt = 0
                    for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and empty[ny][nx]:
                            cnt += 1
                    if cnt % 2 == 0:
                        parity_bonus += 1
                    else:
                        parity_bonus -= 1
        score += parity_bonus * 2
        # X-square, C-square, A-square危険度評価
        x_squares = [(1,1),(1,6),(6,1),(6,6)]
        c_squares = [(0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6)]
        a_squares = [(0,2),(2,0),(0,5),(2,7),(5,0),(7,2),(5,7),(7,5)]
        for (x, y) in x_squares:
            if board[y][x] == player:
                score -= 25
            elif board[y][x] == opponent:
                score += 25
        for (x, y) in c_squares:
            if board[y][x] == player:
                score -= 15
            elif board[y][x] == opponent:
                score += 15
        for (x, y) in a_squares:
            if board[y][x] == player:
                score -= 8
            elif board[y][x] == opponent:
                score += 8
        # モビリティの先読み（2手先・3手先）
        def lookahead_mobility(b, p, depth):
            if depth == 0:
                return len(self.get_valid_moves(b, p))
            moves = self.get_valid_moves(b, p)
            if not moves:
                return 0
            total = 0
            for x, y in moves:
                new_b = [row[:] for row in b]
                self._simulate_move(new_b, 3-p, x, y) # player を 3-p に修正
                total += lookahead_mobility(new_b, p, depth-1) # player を p に修正
            return total
        score += (lookahead_mobility(board, player, 2) - lookahead_mobility(board, opponent, 2))
        # パターンベース評価
        # 辺
        for y in [0, 7]:
            for x in range(BOARD_SIZE):
                if board[y][x] == player:
                    score += 2
                elif board[y][x] == 3-player:
                    score -= 2
        for x in [0, 7]:
            for y in range(BOARD_SIZE):
                if board[y][x] == player:
                    score += 2
                elif board[y][x] == 3-player:
                    score -= 2
        # 角周り
        for (cx, cy) in [(0,0),(0,7),(7,0),(7,7)]:
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx,ny)!=(cx,cy):
                        if board[ny][nx] == player:
                            score -= 3
                        elif board[ny][nx] == 3-player:
                            score += 3
        # 斜め列
        for i in range(BOARD_SIZE):
            if board[i][i] == player:
                score += 1
            elif board[i][i] == 3-player:
                score -= 1
            if board[i][BOARD_SIZE-1-i] == player:
                score += 1
            elif board[i][BOARD_SIZE-1-i] == 3-player:
                score -= 1
        # ノイズ付与による多様性
        score += random.randint(-2, 2)
        return score

    def find_best_move(self, player, max_time=2.0):
        import time
        moves = self.get_valid_moves(self.board, player)
        if not moves:
            return None
        empty_count = sum(row.count(0) for row in self.board)
        if empty_count <= 12:
            max_depth = empty_count
        else:
            max_depth = self.ai_depth
        best_move = None
        best_score = -float('inf')
        start = time.time()
        if max_depth == 1:
            for x, y in moves:
                new_board = [row[:] for row in self.board]
                self._simulate_move(new_board, player, x, y)
                score = self.alpha_beta(new_board, 3 - player, 0, -float('inf'), float('inf'))
                if score > best_score or best_move is None:
                    best_score = score
                    best_move = (x, y)
        else:
            for depth in range(1, max_depth+1):
                for x, y in moves:
                    if time.time() - start > max_time:
                        return best_move
                    new_board = [row[:] for row in self.board]
                    self._simulate_move(new_board, player, x, y)
                    score = self.alpha_beta(new_board, 3 - player, depth-1, -float('inf'), float('inf'))
                    if score > best_score or best_move is None:
                        best_score = score
                        best_move = (x, y)
        return best_move

    def mcts_best_move(self, player, simulations=100):
        from concurrent.futures import ThreadPoolExecutor
        import math

        moves = self.get_valid_moves(self.board, player)
        if not moves:
            return None

        def run_simulation(move):
            try:
                win_count = 0
                sim_per_move = math.ceil(simulations / len(moves))
                for _ in range(sim_per_move):
                    board_copy = [row[:] for row in self.board]
                    self._simulate_move(board_copy, player, move[0], move[1])
                    winner = self.mcts_playout(board_copy, 3-player)
                    if winner == player:
                        win_count += 1
                return (move, win_count)
            except Exception as e:
                print(f"Error in MCTS simulation: {e}")
                return (move, 0)

        try:
            # スレッド数の決定（moves数に応じて適切な数に）
            n_workers = min(self.max_workers, len(moves))
            if n_workers <= 1:  # シリアル実行
                results = [run_simulation(move) for move in moves]
            else:  # 並列実行
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(run_simulation, moves))

            # 最善手の選択
            best_move, best_wins = max(results, key=lambda x: x[1])
            return best_move

        except Exception as e:
            print(f"Error in MCTS parallel processing: {e}")
            # エラー時は最初の合法手を返す
            return moves[0]

    def mcts_playout(self, board, player):
        current = player
        while True:
            moves = self.get_valid_moves(board, current)
            if not moves:
                current = 3 - current
                moves = self.get_valid_moves(board, current)
                if not moves:
                    break
            move = random.choice(moves)
            self._simulate_move(board, current, move[0], move[1])
            current = 3 - current
        # 勝者判定
        black = sum(row.count(1) for row in board)
        white = sum(row.count(2) for row in board)
        if black > white:
            return 1
        elif white > black:
            return 2
        else:
            return 0

    def _simulate_move(self, board, player, x, y):
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

    def board_hash(self, board, player):
        # 盤面の対称性を考慮したハッシュ
        boards = [
            tuple(tuple(row) for row in board),
            tuple(tuple(row[::-1]) for row in board),  # 左右反転
            tuple(tuple(board[y]) for y in range(BOARD_SIZE-1, -1, -1)),  # 上下反転
            tuple(tuple(board[y][::-1]) for y in range(BOARD_SIZE-1, -1, -1)),  # 上下左右反転
            tuple(tuple(board[x][y] for x in range(BOARD_SIZE)) for y in range(BOARD_SIZE)),  # 転置
            tuple(tuple(board[x][y] for x in range(BOARD_SIZE-1, -1, -1)) for y in range(BOARD_SIZE)),  # 転置左右
            tuple(tuple(board[x][y] for x in range(BOARD_SIZE)) for y in range(BOARD_SIZE-1, -1, -1)),  # 転置上下
            tuple(tuple(board[x][y] for x in range(BOARD_SIZE-1, -1, -1)) for y in range(BOARD_SIZE-1, -1, -1)),  # 転置上下左右
        ]
        min_hash = min(boards)
        return (min_hash, player)

    def quiescence(self, board, player, alpha, beta):
        stand_pat = self.evaluate_board(board, player)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        moves = self.get_valid_moves(board, player)
        # 返せる石が多い手だけ探索
        for x, y in moves:
            flip_count = 0
            directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
            for dx, dy in directions:
                nx, ny = x+dx, y+ny
                while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    if board[ny][nx] == 3-player:
                        flip_count += 1
                    elif board[ny][nx] == player or board[ny][nx] == 0:
                        break
                    nx += dx
                    ny += dy
            if flip_count >= 3:  # 荒れた手のみ
                new_board = [row[:] for row in board]
                self._simulate_move(new_board, player, x, y)
                score = -self.quiescence(new_board, 3-player, -beta, -alpha)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha

    def alpha_beta(self, board, player, depth, alpha, beta, pvs=True):
        key = self.board_hash(board, player)
        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry['depth'] >= depth:
                return entry['value']
        moves = self.get_valid_moves(board, player)
        if depth == 0 or not moves:
            return self.quiescence(board, player, alpha, beta)
        # Multi-ProbCut
        if depth >= 4:
            margin = 100
            probcut_beta = beta + margin
            probcut_score = self.alpha_beta(board, player, depth-2, probcut_beta-1, probcut_beta, pvs)
            if probcut_score >= probcut_beta:
                return probcut_score
            probcut_alpha = alpha - margin
            probcut_score = self.alpha_beta(board, player, depth-2, probcut_alpha, probcut_alpha+1, pvs)
            if probcut_score <= probcut_alpha:
                return probcut_score
        # Killer Move/History Heuristic
        killer = self.killer_moves.get((key, depth))
        if killer and killer in moves:
            moves = [killer] + [m for m in moves if m != killer]
        else:
            moves = sorted(moves, key=lambda m: -self.history_heuristic.get((key, m), 0))
        # Null Move Pruning
        if depth >= 2 and len(moves) > 0:
            null_board = [row[:] for row in board]
            null_score = -self.alpha_beta(null_board, 3 - player, depth-2, -beta, -beta+1, pvs)
            if null_score >= beta:
                return null_score
        reduction = 1 if depth >= 3 else 0
        if player == self.current_player:
            value = -float('inf')
            for i, (x, y) in enumerate(moves):
                new_board = [row[:] for row in board]
                self._simulate_move(new_board, player, x, y)
                d = depth-1
                if i >= 2 and reduction:
                    d -= 1
                if pvs and i > 0:
                    v = -self.alpha_beta(new_board, 3 - player, d, -alpha-1, -alpha, pvs)
                    if v > alpha:
                        v = -self.alpha_beta(new_board, 3 - player, d, -beta, -alpha, pvs)
                else:
                    v = -self.alpha_beta(new_board, 3 - player, d, -beta, -alpha, pvs)
                if v > value:
                    value = v
                if value > alpha:
                    alpha = value
                if value >= beta:
                    self.killer_moves[(key, depth)] = (x, y)
                    self.history_heuristic[(key, (x, y))] = self.history_heuristic.get((key, (x, y)), 0) + 1
                    break
            self.transposition_table[key] = {'value': value, 'depth': depth}
            return value
        else:
            value = float('inf')
            for i, (x, y) in enumerate(moves):
                new_board = [row[:] for row in board]
                self._simulate_move(new_board, player, x, y)
                d = depth-1
                if i >= 2 and reduction:
                    d -= 1
                if pvs and i > 0:
                    v = -self.alpha_beta(new_board, 3 - player, d, -alpha-1, -alpha, pvs)
                    if v < beta:
                        v = -self.alpha_beta(new_board, 3 - player, d, -beta, -alpha, pvs)
                else:
                    v = -self.alpha_beta(new_board, 3 - player, d, -beta, -alpha, pvs)
                if v < value:
                    value = v
                if value < beta:
                    beta = value
                if value <= alpha:
                    self.killer_moves[(key, depth)] = (x, y)
                    self.history_heuristic[(key, (x, y))] = self.history_heuristic.get((key, (x, y)), 0) + 1
                    break
            self.transposition_table[key] = {'value': value, 'depth': depth}
            return value

    def is_valid_move(self, x, y):
        return self.is_valid_move_for(self.board, self.current_player, x, y)

    def start_pondering(self):
        # Human手番中にAIが次の手を予測して先読み
        if self.player_types[self.current_player-1] != "Human":
            return
        board_copy = [row[:] for row in self.board]
        player = self.current_player
        moves = self.get_valid_moves(board_copy, player)
        if not moves:
            return
        # 予測：Humanが最善手を打つと仮定
        best_score = -float('inf')
        best_move = None
        for x, y in moves:
            new_board = [row[:] for row in board_copy]
            self._simulate_move(new_board, player, x, y)
            score = self.evaluate_board(new_board, 3 - player)
            if score > best_score:
                best_score = score
                best_move = (x, y)
        if best_move is None:
            return
        def ponder():
            self.ponder_stop.clear()
            # Humanがbest_moveを打った後の局面を先読み
            new_board = [row[:] for row in board_copy]
            self._simulate_move(new_board, player, best_move[0], best_move[1])
            ai_player = 3 - player
            move = self.find_best_move(ai_player)
            if self.ponder_stop.is_set():
                return
            self.ponder_result = move
            self.ponder_move = best_move
        self.ponder_result = None
        self.ponder_move = None
        if self.ponder_thread and self.ponder_thread.is_alive():
            self.ponder_stop.set()
            self.ponder_thread.join()
        self.ponder_thread = threading.Thread(target=ponder)
        self.ponder_thread.start()

    def stop_pondering(self):
        if self.ponder_thread and self.ponder_thread.is_alive():
            self.ponder_stop.set()
            self.ponder_thread.join()
        self.ponder_result = None
        self.ponder_move = None

    def run(self):
        try:
            ai_waiting = False
            ai_last_time = 0
            while True:
                self.draw_board()
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.stop_pondering()
                        pygame.quit()
                        sys.exit()
                    if event.type == MOUSEBUTTONDOWN:
                        mx, my = event.pos
                        board_left = 40
                        board_top = TOP_UI
                        if (board_left <= mx < board_left+BOARD_PIXEL and board_top <= my < board_top+BOARD_PIXEL and not self.game_over):
                            x = (mx - board_left) // CELL_SIZE
                            y = (my - board_top) // CELL_SIZE
                            self.make_move(x, y)
                            self.stop_pondering()
                        for label, rect in self.button_rects.items():
                            if rect.collidepoint(mx, my):
                                if label == "Reset":
                                    self.reset_board()
                                    self.stop_pondering()
                                elif label == "Undo":
                                    self.undo()
                                    self.stop_pondering()
                                elif label == "Redo":
                                    self.redo()
                                    self.stop_pondering()
                                elif label == "Hint":
                                    if not self.hint_mode:
                                        self.best_move = self.find_best_move(self.current_player)
                                    else:
                                        self.best_move = None
                                    self.hint_mode = not self.hint_mode
                                elif label == "DarkMode":
                                    self.dark_mode = not self.dark_mode
                                elif label in ["AlphaBeta", "MCTS"]:
                                    self.ai_mode = label
                        # プレイヤー種別
                        for i in range(2):
                            for j in range(2):
                                rect = self.player_btn_rects[i][j]
                                if rect and rect.collidepoint(mx, my):
                                    self.player_types[i] = ["Human", "AI"][j]
                                    self.best_move = None
                                    ai_waiting = False
                                    self.stop_pondering()
                        if self.slider_rect and self.slider_rect.collidepoint(mx, my):
                            self.slider_drag = True
                    if event.type == pygame.MOUSEBUTTONUP:
                        self.slider_drag = False
                    if event.type == pygame.MOUSEMOTION and self.slider_drag:
                        mx, my = event.pos
                        slider_x = self.slider_rect.x
                        slider_w = self.slider_rect.width
                        v = int(round((mx - slider_x) / slider_w * 14)) + 1
                        v = max(1, min(15, v))
                        self.ai_depth = v
                # Pondering: Human手番中は先読み
                if not self.game_over and self.player_types[self.current_player-1] == "Human":
                    if not self.ponder_thread or not self.ponder_thread.is_alive():
                        self.start_pondering()
                # AI自動手番
                if not self.game_over and self.player_types[self.current_player-1] == "AI":
                    if not ai_waiting:
                        ai_last_time = time.time()
                        ai_waiting = True
                    elif time.time() - ai_last_time >= 0.5:
                        # 先読み結果があれば利用
                        move = None
                        if self.ponder_result and self.ponder_move:
                            # 直前のHuman手が予測通りなら先読み結果を使う
                            last_move = None
                            for y in range(BOARD_SIZE):
                                for x in range(BOARD_SIZE):
                                    if self.board[y][x] != 0:
                                        continue
                                    # 直前の盤面との差分でHuman手を推定
                                    # ここでは省略的に先読みmoveと一致するかだけ判定
                            if self.ponder_move in self.get_valid_moves(self.board, 3 - self.current_player):
                                move = self.ponder_result
                        if not move:
                            move = self.find_best_move(self.current_player)
                        self.make_move(*move)
                        ai_waiting = False
                        self.stop_pondering()
                else:
                    ai_waiting = False
                pygame.display.update()
        except Exception as e:
            import traceback
            print("[ERROR]", e)
            traceback.print_exc()
            input("Press Enter to exit...")

    def self_play_tuning(self, games=10):
        # 簡易的な自己対戦による重み最適化（例示）
        best_weights = None
        best_score = -float('inf')
        for trial in range(5):
            # ランダムに重みを微調整
            weights = [[w + random.randint(-5,5) for w in row] for row in [
                [150, -40, 30, 10, 10, 30, -40, 150],
                [-40, -60, -5, -5, -5, -5, -60, -40],
                [30, -5, 20, 3, 3, 20, -5, 30],
                [10, -5, 3, 3, 3, 3, -5, 10],
                [10, -5, 3, 3, 3, 3, -5, 10],
                [30, -5, 20, 3, 3, 20, -5, 30],
                [-40, -60, -5, -5, -5, -5, -60, -40],
                [150, -40, 30, 10, 10, 30, -40, 150],
            ]]
            score = 0
            for g in range(games):
                self.board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
                self.board[3][3] = self.board[4][4] = 2
                self.board[3][4] = self.board[4][3] = 1
                self.current_player = 1
                for _ in range(60):
                    move = self.find_best_move(self.current_player)
                    if move:
                        self.make_move(*move, animate=False)
                    else:
                        self.current_player = 3 - self.current_player
                        if not self.get_valid_moves(self.board, self.current_player):
                            break
                black = sum(row.count(1) for row in self.board)
                white = sum(row.count(2) for row in self.board)
                if black > white:
                    score += 1
                elif white < black: # elif white > black: を修正
                    score -= 1
            if score > best_score:
                best_score = score
                self.weights = weights # best_weights を self.weights に修正 (仮定)

def eval_move_for_mp(args):
    board, player, x, y, depth, game_params = args
    # game_params: 必要なパラメータ（BOARD_SIZE, ...）を辞書で渡す
    # 必要なメソッドを再実装（またはstaticmethod化）
    # ここでは最低限のロジックのみ例示
    import random
    def _simulate_move(board, player, x, y):
        BOARD_SIZE = game_params['BOARD_SIZE']
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
    # alpha_betaは簡易的に評価値を返す例（本来は完全なロジックをstaticmethod化して渡す必要あり）
    # ここでは乱数で代用
    new_board = [row[:] for row in board]
    _simulate_move(new_board, player, x, y)
    score = random.randint(-100, 100)  # 本来はalpha_beta(new_board, ...)を呼ぶ
    return (score, (x, y))