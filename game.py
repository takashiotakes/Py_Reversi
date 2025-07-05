import pygame
import sys
import time
from pygame.locals import *

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

    def save_state(self):
        # Undo/Redo用に盤面・プレイヤー状態を保存
        import copy
        return (copy.deepcopy(self.board), self.current_player, self.player_types[:], self.dark_mode, self.ai_depth)

    def load_state(self, state):
        self.board, self.current_player, self.player_types, self.dark_mode, self.ai_depth = state
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

    def evaluate_board(self, board, player):
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
        return score

    def find_best_move(self, player):
        moves = self.get_valid_moves(self.board, player)
        if not moves:
            return None
        best_score = -float('inf')
        best_move = None
        for x, y in moves:
            new_board = [row[:] for row in self.board]
            self._simulate_move(new_board, player, x, y)
            score = self.alpha_beta(new_board, 3 - player, self.ai_depth-1, -float('inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = (x, y)
        return best_move

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

    def alpha_beta(self, board, player, depth, alpha, beta):
        moves = self.get_valid_moves(board, player)
        if depth == 0 or not moves:
            return self.evaluate_board(board, player)
        if player == self.current_player:
            value = -float('inf')
            for x, y in moves:
                new_board = [row[:] for row in board]
                self._simulate_move(new_board, player, x, y)
                value = max(value, self.alpha_beta(new_board, 3 - player, depth-1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for x, y in moves:
                new_board = [row[:] for row in board]
                self._simulate_move(new_board, player, x, y)
                value = min(value, self.alpha_beta(new_board, 3 - player, depth-1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def is_valid_move(self, x, y):
        return self.is_valid_move_for(self.board, self.current_player, x, y)

    def run(self):
        try:
            ai_waiting = False
            ai_last_time = 0
            while True:
                self.draw_board()
                for event in pygame.event.get():
                    if event.type == QUIT:
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
                        for label, rect in self.button_rects.items():
                            if rect.collidepoint(mx, my):
                                if label == "Reset":
                                    self.reset_board()
                                elif label == "Undo":
                                    self.undo()
                                elif label == "Redo":
                                    self.redo()
                                elif label == "Hint":
                                    if not self.hint_mode:
                                        self.best_move = self.find_best_move(self.current_player)
                                    else:
                                        self.best_move = None
                                    self.hint_mode = not self.hint_mode
                                elif label == "DarkMode":
                                    self.dark_mode = not self.dark_mode
                        # プレイヤー種別
                        for i in range(2):
                            for j in range(2):
                                rect = self.player_btn_rects[i][j]
                                if rect and rect.collidepoint(mx, my):
                                    self.player_types[i] = ["Human", "AI"][j]
                                    self.best_move = None
                                    ai_waiting = False  # AI割り込み
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
                # AI自動手番
                if not self.game_over and self.player_types[self.current_player-1] == "AI":
                    if not ai_waiting:
                        ai_last_time = time.time()
                        ai_waiting = True
                    elif time.time() - ai_last_time >= 0.5:
                        move = self.find_best_move(self.current_player)
                        if move:
                            self.make_move(*move)
                        ai_waiting = False
                else:
                    ai_waiting = False
                pygame.display.update()
        except Exception as e:
            import traceback
            print("[ERROR]", e)
            traceback.print_exc()
            input("Press Enter to exit...")