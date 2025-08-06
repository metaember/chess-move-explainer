"""Type stubs for the stockfish library."""

from typing import Dict, List, Optional, Union, TypedDict


class EvaluationResult(TypedDict):
    type: str  # "cp" for centipawns or "mate" for mate in X
    value: int  # centipawn value or mate distance


class TopMove(TypedDict):
    Move: str
    Centipawn: Optional[int]
    Mate: Optional[int]


class Stockfish:
    def __init__(
        self,
        path: str = "stockfish",
        depth: int = 18,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None
    ) -> None: ...
    
    def set_fen_position(self, fen: str) -> None:
        """Set the position using FEN notation."""
        ...
    
    def set_position(self, moves: Optional[List[str]] = None) -> None:
        """Set the position from a list of moves in UCI format."""
        ...
    
    def get_best_move(self) -> Optional[str]:
        """Get the best move in UCI format."""
        ...
    
    def get_evaluation(self) -> EvaluationResult:
        """Get the evaluation of the current position."""
        ...
    
    def get_top_moves(self, num_top_moves: int = 5) -> List[TopMove]:
        """Get the top N moves with their evaluations."""
        ...
    
    def set_depth(self, depth: int) -> None:
        """Set the analysis depth."""
        ...
    
    def set_skill_level(self, skill_level: int) -> None:
        """Set skill level (0-20)."""
        ...
    
    def is_fen_valid(self, fen: str) -> bool:
        """Check if a FEN string is valid."""
        ...