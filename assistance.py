# # 3_assistant_ai.py - The Final, AI-Powered, On-Demand Assistant

# import pyautogui
# import cv2
# import numpy as np
# import chess
# import chess.engine
# import os
# import time
# import logging
# import json
# import tensorflow as tf

# # Import settings from our config file
# from config import STOCKFISH_PATH, IMAGE_SIZE, MODEL_PATH, STOCKFISH_THINK_TIME

# # -----------------------------------------------------------------------------
# # Logging (INFO with timestamp + [INFO] + milliseconds)
# # -----------------------------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s -[%(levelname)s]- %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )

# class ChessAIAssistant:
#     """
#     An advanced real-time chess assistant that uses a trained CNN for piece
#     recognition and operates on user demand for maximum stability.
#     """

#     def __init__(self, config):
#         self.config = config
#         self.board_region = None          # ROI (x, y, w, h)
#         self.engine = None
#         self.model = None
#         self.label_map = None
#         self.is_playing_as_black = False  # User's color perception (board flipped)
#         # Live-watching state
#         self._last_pieces_fen = None      # Only the pieces section (no side to move, no castles)
#         self._expected_turn = 'w'         # Whose move we expect next: 'w' or 'b'
#         self._my_side = 'w'               # 'w' if user is White, 'b' if user is Black

#     # -------------------------------------------------------------------------
#     # Setup
#     # -------------------------------------------------------------------------
#     def setup(self):
#         """Runs all necessary setup functions."""
#         return self._load_model_and_labels() and \
#                self._init_stockfish() and \
#                self._select_board_region()

#     def _load_model_and_labels(self):
#         """Loads the trained Keras model and the label map from file."""
#         model_path = self.config['MODEL_PATH']
#         label_map_path = 'models/label_map.json'

#         logging.info(f"Loading AI model from '{model_path}'...")
#         if not os.path.exists(model_path) or not os.path.exists(label_map_path):
#             logging.error("Model or label map not found. Please run the training script first.")
#             return False

#         try:
#             self.model = tf.keras.models.load_model(model_path)
#             with open(label_map_path, 'r') as f:
#                 # Keys are strings in JSON, convert them back to integers for lookup
#                 self.label_map = {int(k): v for k, v in json.load(f).items()}
#             logging.info("AI model and label map loaded successfully.")
#             return True
#         except Exception as e:
#             logging.error(f"Error loading model: {e}")
#             return False

#     def _init_stockfish(self):
#         """Initializes the Stockfish engine."""
#         try:
#             self.engine = chess.engine.SimpleEngine.popen_uci(self.config['STOCKFISH_PATH'])
#             logging.info("Stockfish engine initialized successfully.")
#             return True
#         except Exception as e:
#             logging.error(f"Failed to initialize Stockfish. Ensure 'stockfish.exe' is in the same folder. Details: {e}")
#             return False

#     def _select_board_region(self):
#         """Lets the user select the board region."""
#         logging.info("A window will appear. Draw a TIGHT rectangle on the 8x8 squares only, INSIDE the coordinates.")
#         print("\nSelect a ROI and then press SPACE or ENTER button!\n")
#         print("Cancel the selection process by pressing c button!\n")

#         try:
#             screenshot = pyautogui.screenshot()
#             img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
#             roi = cv2.selectROI("Select Chessboard Region", img, fromCenter=False, showCrosshair=True)
#             cv2.destroyAllWindows()
#             if roi[2] == 0 or roi[3] == 0:
#                 logging.error("No region selected. Exiting.")
#                 return False
#             self.board_region = roi
#             logging.info(f"Capture region selected: {roi}")
#             return True
#         except Exception as e:
#             logging.error(f"Could not select region: {e}")
#             return False

#     # -------------------------------------------------------------------------
#     # Vision → FEN
#     # -------------------------------------------------------------------------
#     def _predict_piece(self, square_img):
#         """Uses the trained CNN model to predict the piece on a square."""
#         resized_square = cv2.resize(square_img, IMAGE_SIZE)
#         normalized_square = resized_square.astype('float32') / 255.0
#         input_tensor = np.expand_dims(normalized_square, axis=0)

#         predictions = self.model.predict(input_tensor, verbose=0)[0]
#         predicted_id = np.argmax(predictions)

#         return self.label_map.get(predicted_id, "empty")

#     def _image_to_fen_pieces(self):
#         """Captures the board and uses the AI model to generate a FEN (pieces only)."""
#         if not self.board_region:
#             return None

#         screenshot = pyautogui.screenshot(region=self.board_region)
#         board_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

#         square_h, square_w = board_img.shape[0] // 8, board_img.shape[1] // 8
#         fen_rows = []
#         for r in range(8):
#             fen_row, empty_count = "", 0
#             for c in range(8):
#                 square = board_img[r*square_h:(r+1)*square_h, c*square_w:(c+1)*square_w]
#                 piece_code = self._predict_piece(square)
#                 if piece_code != "empty":
#                     if empty_count > 0:
#                         fen_row += str(empty_count)
#                     empty_count = 0
#                     piece_char = piece_code[1]
#                     fen_row += piece_char.lower() if piece_code[0] == 'b' else piece_char.upper()
#                 else:
#                     empty_count += 1
#             if empty_count > 0:
#                 fen_row += str(empty_count)
#             fen_rows.append(fen_row)

#         fen = "/".join(fen_rows)
#         # If playing from Black's perspective, flip rows/cols so FEN is in White-bottom convention
#         return '/'.join(row[::-1] for row in fen.split('/')[::-1]) if self.is_playing_as_black else fen

#     # -------------------------------------------------------------------------
#     # Engine helpers
#     # -------------------------------------------------------------------------
#     def _full_fen(self, pieces_fen, turn_char):
#         """Build full FEN string from pieces and side to move."""
#         return f"{pieces_fen} {turn_char} - - 0 1"

#     def _get_best_move(self, fen_pieces, turn_char):
#         """Gets the best move and top-3 from a FEN and turn."""
#         try:
#             full_fen = self._full_fen(fen_pieces, turn_char)
#             board = chess.Board(full_fen)
#             if not board.is_valid():
#                 return "Illegal Position", ""

#             limit = chess.engine.Limit(time=self.config['STOCKFISH_THINK_TIME'])
#             info = self.engine.analyse(board, limit, multipv=3)
#             if not info:
#                 return "No moves found", ""

#             top_moves = []
#             for item in info:
#                 move = item.get('pv', [None])[0]
#                 if move is None:
#                     continue
#                 score = item['score'].pov(board.turn)
#                 eval_str = f"Mate in {score.mate()}" if score.is_mate() else f"{score.score() / 100.0:+.2f}"
#                 top_moves.append(f"{move.uci()} (Eval: {eval_str})")

#             best_move = top_moves[0].split(' ')[0] if top_moves else "NoMove"
#             return best_move, " | ".join(top_moves)
#         except Exception as e:
#             logging.error(f"Engine analysis failed: {e}")
#             return "Analysis Error", ""

#     def _infer_move(self, prev_pieces, next_pieces, mover_turn):
#         """
#         Infer UCI move that transforms prev -> next for the given mover_turn.
#         We compare only piece placement (board_fen).
#         """
#         try:
#             prev = chess.Board(self._full_fen(prev_pieces, mover_turn))
#             # Turn flips after a legal move
#             next_board = chess.Board(self._full_fen(next_pieces, 'b' if mover_turn == 'w' else 'w'))
#             target_boardfen = next_board.board_fen()

#             for mv in prev.legal_moves:
#                 tmp = prev.copy()
#                 tmp.push(mv)
#                 if tmp.board_fen() == target_boardfen:
#                     return mv.uci()
#         except Exception:
#             pass
#         return None

#     # -------------------------------------------------------------------------
#     # Pretty printing
#     # -------------------------------------------------------------------------
#     def _sep(self):
#         print("\n" + "=" * 70 + "\n")

#     # -------------------------------------------------------------------------
#     # Main loop
#     # -------------------------------------------------------------------------
#     def run(self):
#         """The main loop of the chess assistant with the requested UI/flow."""
#         # --- Setup (ROI happens here so its logs appear first) ---
#         if not self.setup():
#             input("Initialization failed. Please check the logs. Press Enter to exit.")
#             return

#         # Ask color in the exact style you want
#         color_input = input("\nAre you playing as Black (board is flipped)? [y/N]: ").strip().lower()
#         self.is_playing_as_black = (color_input == 'y')
#         self._my_side = 'b' if self.is_playing_as_black else 'w'
#         self._expected_turn = 'w'  # At starting position, White to move

#         print("\n\n✅ Assistant ready. Please set up the board to the starting position of your game.")
#         input("\n--> Press ENTER when the starting position is on the screen...")

#         # --- Initial recognition ---
#         current_pieces = self._image_to_fen_pieces()
#         if not current_pieces:
#             print("⚠️ Could not capture the board.")
#             return

#         # Validate initial board quickly (best-effort)
#         try:
#             _ = chess.Board(self._full_fen(current_pieces, self._expected_turn))
#         except ValueError:
#             print(f"⚠️ AI recognized an invalid board state: {current_pieces}")
#             print("This can happen during piece animations. Please try again when the board is still.")
#             return

#         print(f"\n✅ Initial position recognized: {current_pieces}\n")
#         self._last_pieces_fen = current_pieces

#         # --- If it's user's turn, suggest opening move; else just start watching ---
#         my_color_name = "Black" if self._my_side == 'b' else "White"
#         users_turn_now = (self._expected_turn == self._my_side)

#         if users_turn_now:
#             self._sep()
#             print(f"It's your turn to move ({my_color_name}). Analyzing opening move...")
#             best, tops = self._get_best_move(self._last_pieces_fen, self._my_side)
#             print(f"\n♟️ Best Opening Move: {best}")
#             print(f"📊 Top Moves: {tops}")
#             self._sep()
#         else:
#             self._sep()
#             print("Watching for moves...")
#             self._sep()

#         # --- Watcher loop: detect changes, infer moves, analyze when it's our move ---
#         try:
#             while True:
#                 time.sleep(0.25)  # Polling interval

#                 new_pieces = self._image_to_fen_pieces()
#                 if not new_pieces:
#                     continue

#                 if new_pieces != self._last_pieces_fen:
#                     logging.info("Change detected. Analyzing...")
#                     # Whose move did we expect? That side likely made the move.
#                     mover = self._expected_turn  # 'w' or 'b'
#                     move_uci = self._infer_move(self._last_pieces_fen, new_pieces, mover)

#                     if move_uci:
#                         print(f"\n✅ Move Detected: {move_uci}")
#                     else:
#                         print("\n✅ Move Detected: (couldn't infer exact move, continuing)")

#                     # Update board snapshot
#                     self._last_pieces_fen = new_pieces
#                     # Flip expected side to move
#                     self._expected_turn = 'b' if self._expected_turn == 'w' else 'w'

#                     # If opponent just moved, analyze our best reply
#                     if mover != self._my_side:  # Opponent made the move we just detected
#                         print("\nAnalyzing for your best move...")
#                         best, tops = self._get_best_move(self._last_pieces_fen, self._my_side)
#                         print(f"\n♟️ Best Move: {best}")
#                         print(f"📊 Top Moves: {tops}")
#                         self._sep()
#                     else:
#                         # We (user) just moved; now wait for opponent
#                         print("\nOpponent is thinking...")
#                         self._sep()

#         except KeyboardInterrupt:
#             print("\nProgram stopped by user.")
#         finally:
#             if self.engine:
#                 self.engine.quit()


# if __name__ == "__main__":
#     config = {
#         "STOCKFISH_PATH": STOCKFISH_PATH,
#         "MODEL_PATH": MODEL_PATH,
#         "STOCKFISH_THINK_TIME": STOCKFISH_THINK_TIME,
#         "IMAGE_SIZE": IMAGE_SIZE  # Pass image size to the class
#     }
#     assistant = ChessAIAssistant(config)
#     assistant.run()
# 3_assistant_ai.py - The Final, AI-Powered, On-Demand Assistant
import pyautogui
import cv2
import numpy as np
import chess
import chess.engine
import os
import time
import logging
import json
import tensorflow as tf
import threading

# Import settings from our config file
from config import STOCKFISH_PATH, IMAGE_SIZE, MODEL_PATH, STOCKFISH_THINK_TIME

# -----------------------------------------------------------------------------
# Logging (INFO with timestamp + [INFO] + milliseconds)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s -[%(levelname)s]- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class ChessAIAssistant:
    """
    An advanced real-time chess assistant that uses a trained CNN for piece
    recognition and operates on user demand for maximum stability.
    Optimizations included:
      - Batch prediction for 64 squares (single model.predict call)
      - Continuous screenshot thread (non-blocking capture)
      - Fixed downscale to 480x480 for fast cropping/prediction
      - Frame "debounce" via blur detection to skip animated frames
      - Adaptive polling speed based on recognition time
    """

    def __init__(self, config):
        self.config = config
        self.board_region = None          # ROI (x, y, w, h)
        self.engine = None
        self.model = None
        self.label_map = None
        self.is_playing_as_black = False  # User's color perception (board flipped)
        # Live-watching state
        self._last_pieces_fen = None      # Only the pieces section (no side to move, no castles)
        self._expected_turn = 'w'         # Whose move we expect next: 'w' or 'b'
        self._my_side = 'w'               # 'w' if user is White, 'b' if user is Black

        # Optimization / runtime state
        self.latest_frame = None          # populated by screenshot thread
        # poll speed will adapt to workload; start at 0.25s
        self.poll_speed = 0.25
        # image size expected by model (tuple)
        try:
            self.image_size = tuple(self.config.get('IMAGE_SIZE', IMAGE_SIZE))
        except Exception:
            self.image_size = IMAGE_SIZE

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    def setup(self):
        """Runs all necessary setup functions."""
        ok = self._load_model_and_labels() and self._init_stockfish() and self._select_board_region()
        # Start screenshot thread only after board_region is selected successfully
        if ok and self.board_region:
            threading.Thread(target=self._screenshot_thread, daemon=True).start()
            logging.info("Screenshot capture thread started.")
        return ok

    def _load_model_and_labels(self):
        """Loads the trained Keras model and the label map from file."""
        model_path = self.config['MODEL_PATH']
        label_map_path = 'models/label_map.json'

        logging.info(f"Loading AI model from '{model_path}'...")
        if not os.path.exists(model_path) or not os.path.exists(label_map_path):
            logging.error("Model or label map not found. Please run the training script first.")
            return False

        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(label_map_path, 'r') as f:
                # Keys are strings in JSON, convert them back to integers for lookup
                self.label_map = {int(k): v for k, v in json.load(f).items()}
            logging.info("AI model and label map loaded successfully.")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False

    def _init_stockfish(self):
        """Initializes the Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.config['STOCKFISH_PATH'])
            logging.info("Stockfish engine initialized successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize Stockfish. Ensure 'stockfish.exe' is in the same folder. Details: {e}")
            return False

    def _select_board_region(self):
        """Lets the user select the board region."""
        logging.info("A window will appear. Draw a TIGHT rectangle on the 8x8 squares only, INSIDE the coordinates.")
        print("\nSelect a ROI and then press SPACE or ENTER button!\n")
        print("Cancel the selection process by pressing c button!\n")

        try:
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            roi = cv2.selectROI("Select Chessboard Region", img, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            if roi[2] == 0 or roi[3] == 0:
                logging.error("No region selected. Exiting.")
                return False
            self.board_region = roi
            logging.info(f"Capture region selected: {roi}")
            return True
        except Exception as e:
            logging.error(f"Could not select region: {e}")
            return False

    # -------------------------------------------------------------------------
    # Screenshot thread (non-blocking capture)
    # -------------------------------------------------------------------------
    def _screenshot_thread(self):
        """
        Continuously captures screenshots of the selected board_region and stores
        the latest frame in self.latest_frame. This decouples capture from
        prediction and reduces blocking time.
        """
        while True:
            try:
                img = pyautogui.screenshot(region=self.board_region)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # Immediately downscale to target processing resolution for consistency
                frame = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_AREA)
                self.latest_frame = frame
            except Exception:
                # ignore intermittent capture errors; latest_frame remains as last good frame
                pass
            # small sleep to avoid hammering the OS capture; 30ms ~ 33 FPS capture
            time.sleep(0.03)

    # -------------------------------------------------------------------------
    # Blur detection (frame debounce)
    # -------------------------------------------------------------------------
    def _is_blurry(self, img, threshold=80):
        """
        Detect if frame is blurry (piece dragging animation).
        If blurry → skip prediction to avoid errors.
        threshold: variance threshold; lower means stricter blur detection.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var < threshold
        except Exception:
            return True  # If something goes wrong, treat as blurry to be safe

    # -------------------------------------------------------------------------
    # FAST Vision (Batch Prediction)
    # -------------------------------------------------------------------------
    def _predict_batch(self, squares):
        """
        Predicts all N squares in a SINGLE batch.
        squares: list of square images (BGR)
        Returns list of piece labels ("wp", "bp", "empty", etc.)
        """
        if not squares:
            return []

        # Resize/preprocess batch to model IMAGE_SIZE and normalize
        try:
            batch = np.array([
                cv2.resize(sq, self.image_size).astype("float32") / 255.0
                for sq in squares
            ])
            # Single model.predict call for the entire batch
            preds = self.model.predict(batch, verbose=0)
            results = []
            for pred in preds:
                label_id = int(np.argmax(pred))
                results.append(self.label_map.get(label_id, "empty"))
            return results
        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            # Fallback: return empties so we don't crash the loop
            return ["empty"] * len(squares)

    def _image_to_fen_pieces(self):
        """
        MUCH FASTER VERSION
        Uses the latest threaded screenshot -> splits into 64 -> batch predicts -> returns FEN.
        Returns None if no valid frame is available or the frame is blurry.
        """
        # Use the threaded latest_frame to avoid blocking capture
        if self.latest_frame is None:
            return None

        board_img = self.latest_frame.copy()  # already resized to 480x480 in thread

        # Safety check: if too blurry (piece drag/animation), skip this frame
        if self._is_blurry(board_img):
            return None

        try:
            H, W = board_img.shape[:2]
            sq_h, sq_w = H // 8, W // 8

            all_squares = []
            # Extract all 64 squares row-major (top-left -> bottom-right)
            for r in range(8):
                for c in range(8):
                    y1, y2 = r * sq_h, (r + 1) * sq_h
                    x1, x2 = c * sq_w, (c + 1) * sq_w
                    sq = board_img[y1:y2, x1:x2]
                    all_squares.append(sq)

            # Batch predict all squares in one call
            predicted_labels = self._predict_batch(all_squares)

            # Build FEN rows
            fen_rows = []
            idx = 0
            for r in range(8):
                fen_row = ""
                empty_count = 0
                for c in range(8):
                    label = predicted_labels[idx]
                    idx += 1

                    if label != "empty":
                        if empty_count > 0:
                            fen_row += str(empty_count)
                        empty_count = 0
                        piece_char = label[1]  # p,n,b,r,q,k style expected
                        fen_row += piece_char.lower() if label[0] == 'b' else piece_char.upper()
                    else:
                        empty_count += 1

                if empty_count > 0:
                    fen_row += str(empty_count)

                fen_rows.append(fen_row)

            fen = "/".join(fen_rows)

            # If playing from Black's perspective, flip rows/cols so FEN is in White-bottom convention
            if self.is_playing_as_black:
                fen = '/'.join(row[::-1] for row in fen.split('/')[::-1])

            return fen
        except Exception as e:
            logging.error(f"Error building FEN from image: {e}")
            return None

    # -------------------------------------------------------------------------
    # Engine helpers
    # -------------------------------------------------------------------------
    def _full_fen(self, pieces_fen, turn_char):
        """Build full FEN string from pieces and side to move."""
        return f"{pieces_fen} {turn_char} - - 0 1"

    def _get_best_move(self, fen_pieces, turn_char):
        """Gets the best move and top-3 from a FEN and turn."""
        try:
            full_fen = self._full_fen(fen_pieces, turn_char)
            board = chess.Board(full_fen)
            if not board.is_valid():
                return "Illegal Position", ""

            limit = chess.engine.Limit(time=self.config['STOCKFISH_THINK_TIME'])
            info = self.engine.analyse(board, limit, multipv=3)
            if not info:
                return "No moves found", ""

            top_moves = []
            for item in info:
                move = item.get('pv', [None])[0]
                if move is None:
                    continue
                score = item['score'].pov(board.turn)
                eval_str = f"Mate in {score.mate()}" if score.is_mate() else f"{score.score() / 100.0:+.2f}"
                top_moves.append(f"{move.uci()} (Eval: {eval_str})")

            best_move = top_moves[0].split(' ')[0] if top_moves else "NoMove"
            return best_move, " | ".join(top_moves)
        except Exception as e:
            logging.error(f"Engine analysis failed: {e}")
            return "Analysis Error", ""

    def _infer_move(self, prev_pieces, next_pieces, mover_turn):
        """
        Infer UCI move that transforms prev -> next for the given mover_turn.
        We compare only piece placement (board_fen).
        """
        try:
            prev = chess.Board(self._full_fen(prev_pieces, mover_turn))
            # Turn flips after a legal move
            next_board = chess.Board(self._full_fen(next_pieces, 'b' if mover_turn == 'w' else 'w'))
            target_boardfen = next_board.board_fen()

            for mv in prev.legal_moves:
                tmp = prev.copy()
                tmp.push(mv)
                if tmp.board_fen() == target_boardfen:
                    return mv.uci()
        except Exception:
            pass
        return None

    # -------------------------------------------------------------------------
    # Pretty printing
    # -------------------------------------------------------------------------
    def _sep(self):
        print("\n" + "=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    def run(self):
        """The main loop of the chess assistant with the requested UI/flow."""
        # --- Setup (ROI happens here so its logs appear first) ---
        if not self.setup():
            input("Initialization failed. Please check the logs. Press Enter to exit.")
            return

        # Ask color in the exact style you want
        color_input = input("\nAre you playing as Black (board is flipped)? [y/N]: ").strip().lower()
        self.is_playing_as_black = (color_input == 'y')
        self._my_side = 'b' if self.is_playing_as_black else 'w'
        self._expected_turn = 'w'  # At starting position, White to move

        print("\n\n✅ Assistant ready. Please set up the board to the starting position of your game.")
        input("\n--> Press ENTER when the starting position is on the screen...")

        # --- INITIAL RECOGNITION WITH FULL AUTO TURN DETECTION ---

        print("\n🔍 Detecting current board state... Hold still...\n")

        # Step 1 — wait for stable board (no animations)
        while True:
            fen1 = self._image_to_fen_pieces()
            time.sleep(0.6)
            fen2 = self._image_to_fen_pieces()

            if fen1 and fen2 and fen1 == fen2:
                current_pieces = fen1
                break
            else:
                print("⏳ Board moving or animation detected... waiting...")

        print(f"\n📌 Stable board detected:\n{current_pieces}\n")

        # Step 2 — DETERMINE WHOSE TURN IT IS (white or black)
        turn_possibilities = {
            "w": chess.Board(self._full_fen(current_pieces, "w")).is_valid(),
            "b": chess.Board(self._full_fen(current_pieces, "b")).is_valid()
        }

        # Step 3 — decide final turn logically
        if turn_possibilities["w"] and not turn_possibilities["b"]:
            detected_turn = "w"
        elif turn_possibilities["b"] and not turn_possibilities["w"]:
            detected_turn = "b"
        else:
            # Both legal or both illegal — fallback based on your color
            detected_turn = "w" if self._my_side == "w" else "b"

        self._expected_turn = detected_turn

        print(f"🕒 Detected turn: {'White to move' if detected_turn=='w' else 'Black to move'}")

        # Step 4 — save the stable FEN
        self._last_pieces_fen = current_pieces

        # Step 5 — If it's your turn, analyze immediately
        if self._expected_turn == self._my_side:
            my_color_name = "Black" if self._my_side == 'b' else "White"
            self._sep()
            print(f"It's your turn to move ({my_color_name}). Analyzing best move...")
            best, tops = self._get_best_move(self._last_pieces_fen, self._my_side)
            print(f"\n♟️ Best Move: {best}")
            print(f"📊 Top Moves: {tops}")
            self._sep()
        else:
            self._sep()
            print("Watching for opponent's move...")
            self._sep()


        # --- If it's user's turn, suggest opening move; else just start watching ---
        my_color_name = "Black" if self._my_side == 'b' else "White"
        users_turn_now = (self._expected_turn == self._my_side)

        if users_turn_now:
            self._sep()
            print(f"It's your turn to move ({my_color_name}). Analyzing opening move...")
            best, tops = self._get_best_move(self._last_pieces_fen, self._my_side)
            print(f"\n♟️ Best Opening Move: {best}")
            print(f"📊 Top Moves: {tops}")
            self._sep()
        else:
            self._sep()
            print("Watching for moves...")
            self._sep()

        # --- Watcher loop: detect changes, infer moves, analyze when it's our move ---
        try:
            while True:
                time.sleep(self.poll_speed)

                start_time = time.time()
                new_pieces = self._image_to_fen_pieces()
                recognition_time = time.time() - start_time

                # Adaptive polling
                try:
                    if recognition_time > 0.25:
                        self.poll_speed = min(self.poll_speed + 0.05, 0.6)
                    else:
                        self.poll_speed = max(self.poll_speed - 0.03, 0.12)
                except:
                    pass

                if not new_pieces:
                    continue

                # -- DOUBLE CONFIRMATION MOVE DETECTION --
                if new_pieces != self._last_pieces_fen:

                    # First detection, wait a bit
                    time.sleep(0.30)
                    confirm = self._image_to_fen_pieces()

                    # If confirm is None or invalid → ignore
                    if not confirm:
                        continue

                    # If it reverted back → noise, ignore it
                    if confirm == self._last_pieces_fen:
                        continue

                    # If confirm does NOT match new_pieces → still noisy
                    if confirm != new_pieces:
                        continue

                    # --- REAL MOVE DETECTED ---
                    new_pieces = confirm
                    logging.info("Confirmed move detected. Analyzing...")

                    mover = self._expected_turn
                    move_uci = self._infer_move(self._last_pieces_fen, new_pieces, mover)

                    if move_uci:
                        print(f"\n✅ Move Detected: {move_uci}")
                    else:
                        print("\n✅ Move Detected: (couldn't infer exact move, continuing)")

                    self._last_pieces_fen = new_pieces
                    self._expected_turn = 'b' if self._expected_turn == 'w' else 'w'

                    # Opponent moved → give best move
                    if mover != self._my_side:
                        print("\nAnalyzing for your best move...")
                        best, tops = self._get_best_move(self._last_pieces_fen, self._my_side)
                        print(f"\n♟️ Best Move: {best}")
                        print(f"📊 Top Moves: {tops}")
                        self._sep()
                    else:
                        print("\nOpponent is thinking...")
                        self._sep()

        except KeyboardInterrupt:
            print("\nProgram stopped by user.")
        finally:
            if self.engine:
                self.engine.quit()






if __name__ == "__main__":
    config = {
        "STOCKFISH_PATH": STOCKFISH_PATH,
        "MODEL_PATH": MODEL_PATH,
        "STOCKFISH_THINK_TIME": STOCKFISH_THINK_TIME,
        "IMAGE_SIZE": IMAGE_SIZE  # Pass image size to the class
    }
    assistant = ChessAIAssistant(config)
    assistant.run()
