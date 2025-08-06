# Chess Position Analyzer

Analyze chess positions by comparing optimal play versus actual game moves using Stockfish and AI language models.

## Features

- Analyze positions from PGN files, Lichess URLs, or FEN strings
- Compare the best move vs the move actually played
- Get AI-powered explanations of why certain moves are better
- Support for multiple AI models (Claude, GPT-4, etc.)

## Prerequisites

- Python 3.13+
- Stockfish chess engine
- API key for your chosen AI model (Claude, OpenAI, etc.)

## Installation

### Install uv (Python package manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install Stockfish

```bash
# macOS
brew install stockfish

# Ubuntu/Debian
sudo apt-get install stockfish

# Windows
# Download from https://stockfishchess.org/download/
```

### Install project dependencies

```bash
# Clone the repository
git clone <repository-url>
cd chess-explain

# Install dependencies with uv
uv sync
```

### Set up API keys

Create a `.env` file in the project root:

```bash
# For Claude models
ANTHROPIC_API_KEY=your_api_key_here

# For OpenAI models
OPENAI_API_KEY=your_api_key_here
```

## Usage

The tool analyzes chess positions by comparing what was actually played versus the objectively best move.

### Basic syntax

```bash
python chess_analysis.py <input> [move_number] [options]
```

### Examples


** THE METHODS BESIDES FEN + MOVE MIGHT BE BUGGY **

#### 1. Analyze a Lichess game

```bash
# Analyze move 15 from a Lichess game
python chess_analysis.py https://lichess.org/UQuo9KX9 15

# Use a different AI model
python chess_analysis.py https://lichess.org/UQuo9KX9 15 --model claude-3-opus-20240229

# Specify custom Stockfish path
python chess_analysis.py https://lichess.org/UQuo9KX9 15 --stockfish /usr/local/bin/stockfish
```

#### 2. Analyze from a PGN file

```bash
# Analyze move 23 from a PGN file
python chess_analysis.py game.pgn 23

# With more analysis depth
python chess_analysis.py game.pgn 23 --depth 25
```

#### 3. Analyze a FEN position

```bash
python chess_analysis.py "2rrn1k1/1b1qbppp/p3p3/1p1pP3/1n1P4/1P3NNP/P2BQPP1/RB2R1K1 b - - 2 20" Nc2
```

#### 4. Analyze with raw PGN string

```bash
# Pass PGN directly as a string
python chess_analysis.py "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6" 4
```

## Command-line Options

- `--model MODEL`: AI model to use (default: claude-3-5-sonnet-20241022)
  - Claude models: claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-3-sonnet-20240229
  - OpenAI models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
- `--stockfish PATH`: Path to Stockfish executable (default: /opt/homebrew/bin/stockfish)
- `--depth N`: Stockfish analysis depth (default: 20)
- `--n-moves N`: Number of moves to analyze in the continuation (default: 10)

## Output

The tool will:
1. Show the position being analyzed
2. Display the best move vs the actual move played
3. Show the evaluation difference
4. Provide an AI-generated explanation of why the best move is superior
5. List key tactical and positional differences
6. Give a summary of the analysis

## Example Output

```
Analyzing position after move 15...
FEN: r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 8

Move Comparison:
Optimal move: O-O
Actual move: Bxc6

Evaluation difference: 0.75

--- Analysis ---
The optimal move O-O (castling) is superior to Bxc6 for several reasons:

1. King safety: Castling immediately secures the king, moving it away from the center
2. Development: Castling also activates the rook, bringing it into the game
3. Flexibility: Maintaining the bishop preserves more options for future play

Key Differences:
1. After O-O, White maintains the bishop pair advantage
2. The king is safer after castling compared to remaining in the center
3. White retains more dynamic possibilities with the light-squared bishop

Summary: Castling is objectively better as it improves king safety while maintaining piece flexibility
```

## Troubleshooting

1. **Stockfish not found**: Make sure Stockfish is installed and the path is correct
2. **API key errors**: Ensure your `.env` file contains valid API keys
3. **Invalid FEN**: Check that FEN strings are properly formatted
4. **Lichess connection issues**: Verify the game URL is correct and publicly accessible