import pygame
import sys
from pygame.locals import *

# 色の定義
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)

# ボードのサイズ
BOARD_SIZE = 8
CELL_SIZE = 60
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE

class ReversiGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Reversi')
        
        # ボードの初期化 (0: 空, 1: 黒, 2: 白)
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        # 初期配置
        self.board[3][3] = self.board[4][4] = 2  # 白
        self.board[3][4] = self.board[4][3] = 1  # 黒
        
        self.current_player = 1  # 黒から開始
        self.game_over = False

    def draw_board(self):
        self.screen.fill(GREEN)
        
        # グリッドの描画
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                pygame.draw.rect(self.screen, BLACK, 
                               (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
                
                # 石の描画
                if self.board[y][x] > 0:
                    color = BLACK if self.board[y][x] == 1 else WHITE
                    center = (x * CELL_SIZE + CELL_SIZE // 2,
                            y * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(self.screen, color, center, CELL_SIZE // 2 - 4)

        # 有効な手の表示
        valid_moves = self.get_valid_moves()
        for x, y in valid_moves:
            center = (x * CELL_SIZE + CELL_SIZE // 2,
                     y * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, BLUE, center, 5)

        pygame.display.flip()

    def is_valid_move(self, x, y):
        if self.board[y][x] != 0:
            return False

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]

        for dx, dy in directions:
            if self._can_flip(x, y, dx, dy):
                return True
        return False

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

    def make_move(self, x, y):
        if not self.is_valid_move(x, y):
            return False

        self.board[y][x] = self.current_player
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]

        for dx, dy in directions:
            if self._can_flip(x, y, dx, dy):
                self._flip_stones(x, y, dx, dy)

        self.current_player = 3 - self.current_player
        if not self.has_valid_moves():
            self.current_player = 3 - self.current_player
            if not self.has_valid_moves():
                self.game_over = True
        return True

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

    def get_valid_moves(self):
        moves = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.is_valid_move(x, y):
                    moves.append((x, y))
        return moves

    def has_valid_moves(self):
        return len(self.get_valid_moves()) > 0

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == MOUSEBUTTONDOWN and not self.game_over:
                    x, y = event.pos[0] // CELL_SIZE, event.pos[1] // CELL_SIZE
                    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                        self.make_move(x, y)

            self.draw_board()

if __name__ == '__main__':
    game = ReversiGame()
    game.run()
