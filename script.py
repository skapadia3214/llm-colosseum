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
        frame_shape=[301, 512, 0],
        player_1=Player1(
            nickname="Groq Gemma-7B-it",
            model="groq:gemma-7b-it",
        ),
        player_2=Player2(
            nickname="Sambanova Gemma-7B-it",
            model="samba:google/gemma-7b-it",
        ),
        # player_2=Player2(
            # nickname="OpenAI GPT-4o",
            # model="openai:gpt-4o",
        # ),
    )
    game.run()


if __name__ == "__main__":
    main()
