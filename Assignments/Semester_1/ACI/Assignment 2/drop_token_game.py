import random
import time

# ==================== Game Configuration ==========================

BOARD_ROWS = 5                 # board with 5 rows
BOARD_COLUMNS = 5                 # board with 5 columns
SEARCH_DEPTH = 3               # fixed depth for minimax search

EMPTY_CELL = 0                 # empty cell
PLAYER_HUMAN = -1              # human player's token (H)
PLAYER_AI = +1                 # computer player's token (A)


# Heuristic Scoring Weights
CENTRE_WEIGHT = 2              # reward for placing tokens near the centre (helps long-term strategy)
WIN_NEXT_PLAY_BONUS = 100000    # reward for hvaing a win on next drop, we give very big score
THREE_WIN_SCORE = 100          # reward for having 3 in a row during evaluation (not a real win yet)
TWO_PLAYABLE_SCORE = 12        # reward for having 2 in a row with one empty that can be dropped now
TWO_UNPLAYABLE_SCORE = 3       # small reward for 2 in a row with one empty that is not playable yet (floating)
ONE_SCORE = 1                  # very small reward for a single token in a possible line



# ====================== Board creation & display ======================

# Return a 5x5 empty board
def new_empty_board():
    return [[EMPTY_CELL for _ in range(BOARD_COLUMNS)] for _ in range(BOARD_ROWS)]

# Display the board
def display_board(board):
    symbols = {EMPTY_CELL: ' ', PLAYER_HUMAN: 'H', PLAYER_AI: 'A'}
    line = '-' * (BOARD_COLUMNS * 4 + 1)

    print('\n' + line)
    for row in range(BOARD_ROWS):
        row_str = ''.join(f'| {symbols[board[row][column]]} ' for column in range(BOARD_COLUMNS)) + '|'
        print(row_str)
        print(line)

    # Display columns 1 to 5, so a person can type column number
    print('  ' + '   '.join(str(i) for i in range(1, BOARD_COLUMNS + 1)))


# ====================== Game Board Operations ============================

# Returns a list of columns that are not full (top cell is empty).
def list_legal_columns(board):
  return [column for column in range(BOARD_COLUMNS) if board[0][column] == EMPTY_CELL]


# Returns the lowest empty row in the given column (starts from bottom), or None if the column is full.
def find_lowest_empty_row(board, column):
    for row in range(BOARD_ROWS - 1, -1, -1):
        if board[row][column] == EMPTY_CELL:
            return (row, column)
    return None


# Place the player's token in the chosen column and return the new board and return None if column is full.
def place_token(board, column, player):
    pos = find_lowest_empty_row(board, column)
    if pos is None:
        return None
    new_board = [row[:] for row in board]
    row, column = pos
    new_board[row][column] = player
    return new_board


# ====================== Win/Draw Checks =====================================

# Check if the player has three tokens in a row horizontally, vertically, or diagonally.
def has_player_won(board, player):
    # horizontal
    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS - 2):
            if board[row][column] == board[row][column + 1] == board[row][column + 2] == player:
                return True
    # vertical
    for column in range(BOARD_COLUMNS):
        for row in range(BOARD_ROWS - 2):
            if board[row][column] == board[row + 1][column] == board[row + 2][column] == player:
                return True
    # diagonal (\ -> left aligned)
    for row in range(BOARD_ROWS - 2):
        for column in range(BOARD_COLUMNS - 2):
            if board[row][column] == board[row + 1][column + 1] == board[row + 2][column + 2] == player:
                return True
    # diagonal (/ -> right aligned)
    for row in range(2, BOARD_ROWS):
        for column in range(BOARD_COLUMNS - 2):
            if board[row][column] == board[row - 1][column + 1] == board[row - 2][column + 2] == player:
                return True
    return False


# Returns True if the game has ended (someone won or it is a draw i.e., board is full).
def is_game_over(board):
    return has_player_won(board, PLAYER_HUMAN) or has_player_won(board, PLAYER_AI) or len(list_legal_columns(board)) == 0


# Returns True if the board has no tokens placed yet.
def is_board_empty(board):
    for row in board:
        for cell in row:
            if cell != EMPTY_CELL:
                return False
    return True


# ====================== Heuristic Helper Functions ==================

# Returns True if a token can be dropped in this cell right now.
def is_cell_playable(board, row, column):
    if board[row][column] != EMPTY_CELL:
        return False
    return (row == BOARD_ROWS - 1) or (board[row + 1][column] != EMPTY_CELL)


# Returns how many moves can give an instant win for this player.
# Helps in quickly detecting 'win in one move' situations.
def count_immediate_wins(board, player):
    count = 0
    for column in list_legal_columns(board):
        child = place_token(board, column, player)
        if has_player_won(child, player):
            count += 1
    return count


# Returns a small bonus score for placing tokens the middle row, column, and centre cell.
def score_center_positions(board, player):
    score = 0
    center_cell = (2, 2) # for a 5x5 board, centre is at (2,2)
    middle_row = 2
    middle_column = 2

    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS):
            if board[row][column] == player:
                if (row, column) == center_cell:
                    score += 2 * CENTRE_WEIGHT
                if row == middle_row:
                    score += CENTRE_WEIGHT
                if column == middle_column:
                    score += CENTRE_WEIGHT
    return score


# ====================== Static Evaluation ==============================

# Gives a score for the board from AI's perspective.
# Higher score means good for AI. Lower score means good for Human.
def evaluate_board_state(board, player_view, is_ai_turn_next=None):
    opponent = PLAYER_HUMAN if player_view == PLAYER_AI else PLAYER_AI
    total_score = 0

    # Check if next player can win immediately (big reward or penalty)
    if is_ai_turn_next is not None:
        if is_ai_turn_next:
            next_win_moves = count_immediate_wins(board, PLAYER_AI)
            if next_win_moves > 0:
                return WIN_NEXT_PLAY_BONUS * next_win_moves
        else:
            next_win_moves = count_immediate_wins(board, opponent)
            if next_win_moves > 0:
                return -WIN_NEXT_PLAY_BONUS * next_win_moves

	# Helper function to calculate score for a group of 3 connected cells
    def score_three_cell_window(cells):
        nonlocal total_score
        values = [board[row][column] for (row, column) in cells]
        ai_count = values.count(PLAYER_AI)
        opp_count = values.count(opponent)
        empty_count = values.count(EMPTY_CELL)

        # Score windows that have only AI or only opponent tokens
        if opp_count == 0:
            if ai_count == 3:
                total_score += THREE_WIN_SCORE
            elif ai_count == 2 and empty_count == 1:
                er, ec = next((pos for pos, val in zip(cells, values) if val == EMPTY_CELL))
                if is_cell_playable(board, er, ec):
                    total_score += TWO_PLAYABLE_SCORE
                else:
                    total_score += TWO_UNPLAYABLE_SCORE
            elif ai_count == 1 and empty_count == 2:
                total_score += ONE_SCORE

        if ai_count == 0:
            if opp_count == 3:
                total_score -= THREE_WIN_SCORE
            elif opp_count == 2 and empty_count == 1:
                er, ec = next((pos for pos, val in zip(cells, values) if val == EMPTY_CELL))
                if is_cell_playable(board, er, ec):
                    total_score -= TWO_PLAYABLE_SCORE
                else:
                    total_score -= TWO_UNPLAYABLE_SCORE
            elif opp_count == 1 and empty_count == 2:
                total_score -= ONE_SCORE

    # Check all 3-cell combinations horizontally, vertically, and diagonally
    # Horizontal windows of 3
    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS - 2):
            score_three_cell_window([(row, column + i) for i in range(3)])
    # Vertical windows of 3
    for column in range(BOARD_COLUMNS):
        for row in range(BOARD_ROWS - 2):
            score_three_cell_window([(row + i, column) for i in range(3)])
    # Diagonal "\" windows of 3
    for row in range(BOARD_ROWS - 2):
        for column in range(BOARD_COLUMNS - 2):
            score_three_cell_window([(row + i, column + i) for i in range(3)])
    # Diagonal "/" windows of 3
    for row in range(2, BOARD_ROWS):
        for column in range(BOARD_COLUMNS - 2):
            score_three_cell_window([(row - i, column + i) for i in range(3)])

    # Adding a small bonus for centre control
    total_score += score_center_positions(board, PLAYER_AI)
    total_score -= score_center_positions(board, opponent)

    return total_score


# ================= Minimax algorithm with alpha-beta pruning ==============

# Returns (best_column, best_score), where best_score is evaluated from AI's perspective.
def minimax_alpha_beta(board, depth, alpha, beta, ai_turn):
    legal_columns = list_legal_columns(board)
    game_over = is_game_over(board)

    # Stop search if game ended or depth limit reached
    if depth == 0 or game_over:
        if game_over:
            if has_player_won(board, PLAYER_AI):
                return (None, float('inf'))				# AI wins
            elif has_player_won(board, PLAYER_HUMAN):
                return (None, float('-inf'))			# Human wins
            else:
                return (None, 0)  						# Draw
        else:
            # Pass who moves next so evaluation knows whose turn it is
            return (None, evaluate_board_state(board, PLAYER_AI, is_ai_turn_next=ai_turn))

    # AI's turn: try to maximise score
    if ai_turn:
        best_score = float('-inf')
        best_column = random.choice(legal_columns)  # default in case of tie
        for column in legal_columns:
            child_board = place_token(board, column, PLAYER_AI)
            _, new_score = minimax_alpha_beta(child_board, depth - 1, alpha, beta, False)
            if new_score > best_score:
                best_score = new_score
                best_column = column
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # pruning (stop exploring this branch)
        return best_column, best_score

    # Human's turn: try to minimise score
    else:
        best_score = float('inf')
        best_column = random.choice(legal_columns)
        for column in legal_columns:
            child_board = place_token(board, column, PLAYER_HUMAN)
            _, new_score = minimax_alpha_beta(child_board, depth - 1, alpha, beta, True)
            if new_score < best_score:
                best_score = new_score
                best_column = column
            beta = min(beta, best_score)
            if alpha >= beta:
                break  # pruning (stop exploring this branch)
        return best_column, best_score


# ====================== Turn Handlers ==================================

# Handles AI's turn by checking win, block, or using minimax for best move.
def ai_turn_handler(board):
    if is_game_over(board):
        return board

    print("\n AI's turn...")
    display_board(board)

    # If AI can win immediately, play that move
    for column in list_legal_columns(board):
        next_board = place_token(board, column, PLAYER_AI)
        if has_player_won(next_board, PLAYER_AI):
            time.sleep(0.3)
            return next_board

    # If Human can win next turn, block that move
    for column in list_legal_columns(board):
        next_board = place_token(board, column, PLAYER_HUMAN)
        if has_player_won(next_board, PLAYER_HUMAN):
            time.sleep(0.3)
            return place_token(board, column, PLAYER_AI)

    # Otherwise, use minimax. If board is empty, choose centre first
    if is_board_empty(board):
        centre_column = BOARD_COLUMNS // 2
        if board[BOARD_ROWS - 1][centre_column] == EMPTY_CELL:
            chosen_column = centre_column
        else:
            chosen_column = random.choice(list_legal_columns(board))
    else:
        chosen_column, _ = minimax_alpha_beta(board, SEARCH_DEPTH, float('-inf'), float('inf'), True)

    new_board = place_token(board, chosen_column, PLAYER_AI)
    time.sleep(0.3)
    return new_board


# Handles Human's turn by safely taking and validating user input.
def human_turn_handler(board):
    if is_game_over(board):
        return board

    display_board(board)
    while True:
        try:
            user_input = input("Your move: Choose a column between 1 to 5: ").strip()
            if user_input.lower() == 'q':
                print("Exiting the game. Please restart to play again.")
                exit()
            column = int(user_input) - 1  # user enters 1-5, we convert to 0-4
            if column < 0 or column >= BOARD_COLUMNS:
                print("Please enter a number between 1 and 5.")
                continue
            if board[0][column] != EMPTY_CELL:
                print("Column " + str(column + 1) + " is full. Please choose another column.")
                continue
            return place_token(board, column, PLAYER_HUMAN)
        except ValueError:
            print("Invalid input. Type a number 1..5 or 'q' to quit.")


# ====================== Main Game Loop ==============================

# Runs the Drop Token game until someone wins, it's a draw or the game is stopped.
def play_drop_token_game():
    # Create a fresh 5x5 empty board
    board = new_empty_board()
    print("Welcome to 5x5 Drop Token (3-in-a-row wins).")

    # Ask the player if they want to start first
    choice = ''
    while choice not in ('Y', 'N'):
        print("\nEnter 'y' to play first, 'n' to let AI start, or 'q' to quit.")
        choice = input("Do you want to start first? (y/n/q): ").strip().upper()
        if choice.lower() == 'q':
            print("Exiting the game. Please restart to play again.")
            exit()
        if choice not in ('Y', 'N'):
            print("Please enter 'y' or 'n'.")

    current_turn = PLAYER_HUMAN if choice == 'Y' else PLAYER_AI

    while not is_game_over(board):
        if current_turn == PLAYER_HUMAN:
            board = human_turn_handler(board)
            current_turn = PLAYER_AI
        else:
            board = ai_turn_handler(board)
            current_turn = PLAYER_HUMAN

    display_board(board)
    if has_player_won(board, PLAYER_HUMAN):
        print("YOU WIN!")
    elif has_player_won(board, PLAYER_AI):
        print("AI WINS!")
    else:
        print("DRAW!")


if __name__ == '__main__':
    try:
        play_drop_token_game()
    except KeyboardInterrupt:
        print("\nGame interrupted. Please restart to play again.")
