import chess
import chess.pgn
from stockfish import Stockfish
from typing import Optional, Tuple, List, Dict, Union, TypedDict
import io
from pydantic_ai import Agent
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-3-7-sonnet-latest"


class MoveComparison(BaseModel):
    """Response model for move comparison analysis."""

    explanation: str
    key_differences: List[str]
    evaluation_summary: str


def get_position_from_lichess(game_id: str, move_spec: str) -> Tuple[chess.Board, Optional[str]]:
    """
    Get board position at a specific move from Lichess game.
    
    Args:
        game_id: Lichess game ID
        move_spec: Move specification (e.g., "15" for white's 15th, "15b" for black's 15th)
    
    Returns:
        Tuple of (board position before the move, next move in SAN)
    """
    import requests
    
    url = f"https://lichess.org/game/export/{game_id}?moves=true"
    response = requests.get(url, headers={"Accept": "application/json"})
    data = response.json()
    
    moves = data["moves"].split()
    board = chess.Board()
    
    # Parse move specification
    if move_spec.endswith('b'):
        # Black's move
        move_number = int(move_spec[:-1])
        moves_to_play = (move_number - 1) * 2 + 1  # +1 for black's move
    else:
        # White's move (default)
        move_number = int(move_spec)
        moves_to_play = (move_number - 1) * 2
    
    # Play moves up to the position before the requested move
    for i in range(min(moves_to_play, len(moves))):
        board.push_san(moves[i])
    
    # Get the next move if it exists
    next_move_san = moves[moves_to_play] if moves_to_play < len(moves) else None
    
    return board, next_move_san


class MoveInfo(TypedDict):
    ply: int
    move: str
    move_san: str
    evaluation: Dict[str, Union[str, int]]
    fen: str
    side_to_move: str


class ComparisonResult(TypedDict):
    initial_position: str
    initial_evaluation: Dict[str, Union[str, int]]
    optimal: List[MoveInfo]
    actual: List[MoveInfo]


def compare_move_sequences(
    board: chess.Board,
    actual_move: chess.Move,
    stockfish_path: str = "/usr/local/bin/stockfish",
    n_moves: int = 10,
    depth: int = 20,
) -> ComparisonResult:
    """
    Compare optimal play vs actual game continuation.

    Args:
        board: Current board position
        actual_move: The move actually played in the game
        stockfish_path: Path to Stockfish executable
        n_moves: Number of moves to analyze after the first move
        depth: Stockfish analysis depth

    Returns:
        Dictionary with 'optimal' and 'actual' sequences
    """
    stockfish = Stockfish(path=stockfish_path)
    stockfish.set_depth(depth)

    # Get optimal first move
    stockfish.set_fen_position(board.fen())
    optimal_first_move_uci = stockfish.get_best_move()
    assert optimal_first_move_uci is not None
    optimal_first_move = chess.Move.from_uci(optimal_first_move_uci)

    # Analyze optimal line
    optimal_board = board.copy()
    optimal_board.push(optimal_first_move)

    # Get initial evaluation before moves
    stockfish.set_fen_position(board.fen())
    initial_eval = stockfish.get_evaluation()
    assert initial_eval is not None

    # Optimal sequence starting with best move
    optimal_sequence = [
        {
            "ply": 0,
            "move": optimal_first_move_uci,
            "move_san": board.san(optimal_first_move),
            "evaluation": stockfish.get_evaluation(),
            "fen": optimal_board.fen(),
            "side_to_move": "white" if optimal_board.turn else "black",
        }
    ]

    # Continue with optimal play
    temp_board = optimal_board.copy()
    for ply in range(1, n_moves + 1):
        stockfish.set_fen_position(temp_board.fen())
        best_move_uci = stockfish.get_best_move()
        if not best_move_uci:
            break

        move = chess.Move.from_uci(best_move_uci)
        move_san = temp_board.san(move)
        temp_board.push(move)

        stockfish.set_fen_position(temp_board.fen())
        evaluation = stockfish.get_evaluation()

        optimal_sequence.append(
            {
                "ply": ply,
                "move": best_move_uci,
                "move_san": move_san,
                "evaluation": evaluation,
                "fen": temp_board.fen(),
                "side_to_move": "white" if temp_board.turn else "black",
            }
        )

        if temp_board.is_game_over():
            break

    # Analyze actual game line
    actual_sequence = []
    actual_board = board.copy()
    actual_move_san = actual_board.san(actual_move)
    actual_board.push(actual_move)

    stockfish.set_fen_position(actual_board.fen())
    actual_first_eval = stockfish.get_evaluation()

    actual_sequence = [
        {
            "ply": 0,
            "move": actual_move.uci(),
            "move_san": actual_move_san,
            "evaluation": actual_first_eval,
            "fen": actual_board.fen(),
            "side_to_move": "white" if actual_board.turn else "black",
        }
    ]

    # Continue with optimal play after actual move
    temp_board = actual_board.copy()
    for ply in range(1, n_moves + 1):
        stockfish.set_fen_position(temp_board.fen())
        best_move_uci = stockfish.get_best_move()
        if not best_move_uci:
            break

        move = chess.Move.from_uci(best_move_uci)
        move_san = temp_board.san(move)
        temp_board.push(move)

        stockfish.set_fen_position(temp_board.fen())
        evaluation = stockfish.get_evaluation()

        actual_sequence.append(
            {
                "ply": ply,
                "move": best_move_uci,
                "move_san": move_san,
                "evaluation": evaluation,
                "fen": temp_board.fen(),
                "side_to_move": "white" if temp_board.turn else "black",
            }
        )

        if temp_board.is_game_over():
            break

    return ComparisonResult(
        initial_position=board.fen(),
        initial_evaluation=initial_eval,
        optimal=optimal_sequence,
        actual=actual_sequence,
    )


async def analyze_move_comparison(
    comparison_data: ComparisonResult, model: str = DEFAULT_MODEL
) -> MoveComparison:
    """
    Use an LLM to analyze why the optimal move is better than the actual move.

    Args:
        comparison_data: Output from compare_move_sequences
        model: LLM model to use

    Returns:
        MoveComparison with analysis
    """
    agent = Agent(
        model=model,
        output_type=MoveComparison,
        system_prompt="""You are a chess analysis expert. You will be given two sequences of moves:
1. A sequence starting with the objectively best move followed by optimal play
2. A sequence starting with the move actually played in the game followed by optimal play

Your task is to explain why the optimal first move is better than the actual move played.
Focus on concrete positional and tactical differences that arise from the choice of first move.""",
    )

    # Format the sequences for the LLM
    optimal_moves = [
        f"{i+1}. {m['move_san']} (eval: {m['evaluation']['value']/100:.2f})"
        for i, m in enumerate(comparison_data["optimal"])
    ]
    actual_moves = [
        f"{i+1}. {m['move_san']} (eval: {m['evaluation']['value']/100:.2f})"
        for i, m in enumerate(comparison_data["actual"])
    ]

    prompt = f"""Initial position: {comparison_data['initial_position']}
Initial evaluation: {comparison_data['initial_evaluation']['value']/100:.2f} pawns

Optimal line (starting with best move):
{' '.join(optimal_moves)}

Actual game line (starting with move played):
{' '.join(actual_moves)}

The optimal first move was {comparison_data['optimal'][0]['move_san']},
but the actual move played was {comparison_data['actual'][0]['move_san']}.

Explain why the optimal move is superior, focusing on:
1. The immediate tactical or positional advantages
2. How these advantages manifest in the subsequent play
3. The key differences in the resulting positions after 10 moves"""

    result = await agent.run(prompt)
    return result.output


async def analyze_lichess_position(
    url: str,
    move_spec: str,
    stockfish_path: str = "/usr/local/bin/stockfish",
    model: str = DEFAULT_MODEL,
) -> None:
    """
    Analyze a position from Lichess game.
    """
    import re
    
    match = re.search(r"lichess\.org/([a-zA-Z0-9]{8})", url)
    if not match:
        print("Error: Invalid Lichess URL")
        return
    
    game_id = match.group(1)
    board, next_move_san = get_position_from_lichess(game_id, move_spec)
    
    if not next_move_san:
        print(f"Error: Move {move_spec} not found in the game")
        return
    
    # Convert to FEN + move analysis
    await analyze_fen_with_move(board.fen(), next_move_san, stockfish_path, model)


async def analyze_fen_with_move(
    fen: str,
    move_str: str,
    stockfish_path: str = "/usr/local/bin/stockfish",
    model: str = DEFAULT_MODEL,
) -> None:
    """
    Analyze a specific move in a FEN position.
    """
    board = chess.Board(fen)
    
    try:
        move = board.parse_san(move_str)
        print(f"Analyzing move {move_str} in position:")
        print(f"FEN: {board.fen()}")
        print(board.unicode(borders=True, empty_square=" "))
        
        comparison = compare_move_sequences(board, move, stockfish_path)
        
        print(f"\nOptimal move: {comparison['optimal'][0]['move_san']}")
        print(f"Given move: {comparison['actual'][0]['move_san']}")
        
        eval_diff = comparison['optimal'][0]['evaluation']['value'] - comparison['actual'][0]['evaluation']['value']
        print(f"Evaluation difference: {eval_diff/100:.2f} pawns")
        
        analysis = await analyze_move_comparison(comparison, model=model)
        
        print("\n--- Analysis ---")
        print(analysis.explanation)
        print("\nKey differences:")
        for diff in analysis.key_differences:
            print(f"- {diff}")
        print(f"\nSummary: {analysis.evaluation_summary}")
    except ValueError:
        print(f"Error: Invalid move '{move_str}' for the given position")


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Analyze chess positions")
    parser.add_argument(
        "input", help="FEN string or Lichess URL"
    )
    parser.add_argument(
        "move", help="Move (e.g., 'e4', 'Nf3') for FEN, or move number (e.g., '15', '15b') for Lichess"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="AI model to use")
    parser.add_argument(
        "--stockfish", default="/opt/homebrew/bin/stockfish", help="Path to stockfish"
    )

    args = parser.parse_args()

    async def main() -> None:
        if "/" in args.input and " " in args.input:
            # FEN string
            await analyze_fen_with_move(args.input, args.move, args.stockfish, args.model)
        elif "lichess.org" in args.input:
            # Lichess URL
            await analyze_lichess_position(args.input, args.move, args.stockfish, args.model)
        else:
            print("Error: Input must be either a FEN string or a Lichess URL")
            print("Examples:")
            print('  python chess_analysis.py "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" e4')
            print('  python chess_analysis.py https://lichess.org/abc123 15    # White\'s 15th move')
            print('  python chess_analysis.py https://lichess.org/abc123 15b   # Black\'s 15th move')

    asyncio.run(main())
