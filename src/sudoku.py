import random
import numpy as np

class Sudoku:
    def __init__(self, rows=9, cols=9, board=None):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.generate_board()

        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            self.generate_board()
        else:
            self.board = board

    def generate_board(self):
        """Generate a complete valid Sudoku board using backtracking"""
        
        empty_cell = self._find_empty_cell()
        
        # Base case: if no empty cells, board is complete
        if not empty_cell:
            return True
        
        row, col = empty_cell
        
        # Create randomized list of numbers 1-9 to try
        numbers = list(range(1, 10))
        np.random.shuffle(numbers)
        
        # Try each number in random order
        for num in numbers:
            if self._is_valid(self.board, row, col, num):
                # Place the number
                self.board[row, col] = num
                
                # Recursively try to fill the rest
                if self.generate_board():  # Don't pass board - using global
                    return True 
                
                # Backtrack: remove the number if it doesn't lead to solution
                self.board[row, col] = 0
        
        # If no number works, return False to trigger backtracking
        return False   

    def remove_numbers(self, difficulty=0.5):
        """Remove numbers from the board based on difficulty"""
        total_cells = self.rows * self.cols
        num_to_remove = int(total_cells * difficulty)
        
        for _ in range(num_to_remove):
            row, col = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            while self.board[row, col] == 0:
                row, col = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            self.board[row, col] = 0

    def _is_valid(self, board, row, col, num):
        if num in board[row]:
            return False
        if num in board[:, col]:
            return False
        if num in board[row // 3 * 3:row // 3 * 3 + 3, col // 3 * 3:col // 3 * 3 + 3]:
            return False
        return True
    
    def _find_empty_cell(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row, col] == 0:
                    return (row, col)
        return None

    def print_board(self):
        """Pretty print the board"""
        for i in range(self.rows):
            if i % 3 == 0 and i != 0:
                print("- - - + - - - + - - -")
            for j in range(self.cols):
                if j % 3 == 0 and j != 0:
                    print("| ", end="")
                if j == 8:
                    print(self.board[i, j])
                else:
                    print(str(self.board[i, j]) + " ", end="")


    def _solver(self):
        empty_cell = self._find_empty_cell()

        if empty_cell is None:
            return True

        row, col = empty_cell

        for num in range(1,10):
            if self._is_valid(self.board, row, col, num):
                self.board[row,col] = num

                if self._solver():
                    return True
                
                self.board[row,col] = 0

        return False
        
    def solve_board(self):
        self._solver()

        self.print_board()


if __name__ == "__main__":

    easy_board = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    easy_solution = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])

    sudoku = Sudoku(board=easy_board)
    # print(sudoku._is_valid(sudoku.board, 0, 0, 1))
    # sudoku.print_board()


    sudoku.solve_board()