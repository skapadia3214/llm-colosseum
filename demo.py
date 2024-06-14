import sys

from dotenv import load_dotenv
from eval.game import Game, Player1, Player2
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

load_dotenv()


def main():
    # Environment Settings

    game = Game(
        render=True,
        save_game=True,
        splash_screen=True,
        player_1=Player1(
            nickname="Groq Gemma-7B",
            model="groq:gemma-7b-it",
        ),
        player_2=Player2(
            nickname="OpenAI GPT-3.5-Turbo",
            model="openai:gpt-3.5-turbo",
        ),
    )

    game.run()
    return 0


if __name__ == "__main__":
    main()
