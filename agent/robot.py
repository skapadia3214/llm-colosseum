import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Literal, Optional

import requests
from llama_index.core.base.llms.types import ChatResponse
import json
import numpy as np
from gymnasium import spaces
from loguru import logger
from llama_index.core.llms import ChatMessage
from rich import print

from .config import (
    INDEX_TO_MOVE,
    META_INSTRUCTIONS,
    META_INSTRUCTIONS_WITH_LOWER,
    MOVES,
    NB_FRAME_WAIT,
    X_SIZE,
    Y_SIZE,
)
from .observer import detect_position_from_color
from .llm import get_client


class Robot:
    observations: List[Optional[dict]] = None  # memory
    next_steps: List[int]  # action plan
    actions: dict  # actions of the agents during a step of the game
    # actions of the agents during the previous step of the game
    previous_actions: Dict[str, List[int]]
    reward: float  # reward of the agent

    action_space: spaces.Space
    character: Optional[str] = None  # character name
    side: int  # side of the stage where playing: 0 = left, 1 = right
    current_direction: Literal["Left", "Right"]  # current direction facing
    sleepy: Optional[bool] = False  # if the robot is sleepy
    only_punch: Optional[bool] = False  # if the robot only punch

    model: str  # model of the robot
    super_bar_own: int
    player_nb: int  # player number

    def __init__(
        self,
        action_space: spaces.Space,
        character: str,
        side: int,
        character_color: list,
        ennemy_color: list,
        sleepy: bool = False,
        only_punch: bool = False,
        model: str = "mistral:mistral-large-latest",
        player_nb: int = 0,  # 0 means not specified
    ):
        self.action_space = action_space
        self.character = character
        if side == 0:
            self.current_direction = "Right"
        elif side == 1:
            self.current_direction = "Left"

        self.observations = []
        self.next_steps = []
        self.character_color = character_color
        self.ennemy_color = ennemy_color
        self.side = side
        self.sleepy = sleepy
        self.only_punch = only_punch
        self.model = model
        self.previous_actions = defaultdict(list)
        self.actions = {}
        self.player_nb = player_nb
        self.ttft_list = []
        self.reponse_time_list = []
        self.tps_list = []

    def act(self) -> int:
        """
        At each game frame, we execute the first action in the list of next steps.

        An action is an integer from 0 to 18, where 0 is no action.

        See the MOVES dictionary for the mapping of actions to moves.
        """
        if not self.next_steps or len(self.next_steps) == 0:
            return 0  # No move

        if self.sleepy:
            return 0

        if self.only_punch:
            # Do a Hadouken
            if self.current_direction == "Right":
                self.next_steps.extend(
                    [
                        MOVES["Down"],
                        MOVES["Right+Down"],
                        MOVES["Right"],
                        MOVES["High Punch"],
                    ]
                )
            elif self.current_direction == "Left":
                self.next_steps.extend(
                    [
                        MOVES["Down"],
                        MOVES["Down+Left"],
                        MOVES["Left"],
                        MOVES["High Punch"],
                    ]
                )

        next_step = self.next_steps.pop(0)

        return next_step

    def plan(self) -> None:
        """
        The robot will plan its next steps by calling this method.

        In SF3, moves are based on combos, which are list of actions that must be executed in a sequence.

        Moves of Ken
        https://www.eventhubs.com/guides/2009/may/11/ken-street-fighter-3-third-strike-character-guide/

        Moves of Ryu
        https://www.eventhubs.com/guides/2008/may/09/ryu-street-fighter-3-third-strike-character-guide/
        """

        # If we already have a next step, we don't need to plan
        if len(self.next_steps) > 0:
            return

        # Call the LLM to get the next steps
        next_steps_from_llm = self.get_moves_from_llm()
        next_buttons_to_press = [
            button
            for combo in next_steps_from_llm
            for button in META_INSTRUCTIONS_WITH_LOWER[combo][
                self.current_direction.lower()
            ]
            # We add a wait time after each button press
            + [0] * NB_FRAME_WAIT
        ]
        self.next_steps.extend(next_buttons_to_press)

    def observe(self, observation: dict, actions: dict, reward: float):
        """
        The robot will observe the environment by calling this method.

        The latest observations are at the end of the list.
        """

        # detect the position of characters and ennemy based on color
        observation["character_position"] = detect_position_from_color(
            observation, self.character_color
        )
        observation["ennemy_position"] = detect_position_from_color(
            observation, self.ennemy_color
        )

        self.observations.append(observation)
        # we delete the oldest observation if we have more than 10 observations
        if len(self.observations) > 10:
            self.observations.pop(0)

        self.reward = reward

        if actions.get("agent_0") is not None and actions.get("agent_0") != 0:
            self.previous_actions["agent_0"].append(actions["agent_0"])
        if actions.get("agent_1") is not None and actions.get("agent_1") != 0:
            self.previous_actions["agent_1"].append(actions["agent_1"])

        for key, value in actions.items():
            if len(self.previous_actions[key]) > 10:
                self.previous_actions[key].pop(0)

        # Keep track of the current direction by checking the position of the character
        # and the ennemy
        character_position = observation.get("character_position")
        ennemy_position = observation.get("ennemy_position")
        if (
            character_position is not None
            and ennemy_position is not None
            and len(character_position) == 2
            and len(ennemy_position) == 2
        ):
            if character_position[0] < ennemy_position[0]:
                self.current_direction = "Right"
            else:
                self.current_direction = "Left"

    def context_prompt(self) -> str:
        """
        Return a str of the context

        "The observation for you is Left"
        "The observation for the opponent is Left+Up"
        "The action history is Up"
        """

        # Create the position prompt
        side = self.side
        obs_own = self.observations[-1]["character_position"]
        obs_opp = self.observations[-1]["ennemy_position"]
        super_bar_own = self.observations[-1]["P" + str(side + 1)]["super_bar"][0]

        if obs_own is not None and obs_opp is not None:
            relative_position = np.array(obs_own) - np.array(obs_opp)
            normalized_relative_position = [
                relative_position[0] / X_SIZE,
                relative_position[1] / Y_SIZE,
            ]
        else:
            normalized_relative_position = [0.3, 0]

        position_prompt = ""
        if abs(normalized_relative_position[0]) > 0.1:
            position_prompt += (
                "You are very far from the opponent. Move closer to the opponent."
            )
            if normalized_relative_position[0] < 0:
                position_prompt += "Your opponent is on the right."
            else:
                position_prompt += "Your opponent is on the left."

        else:
            position_prompt += "You are close to the opponent. You should attack him."

        power_prompt = ""
        if super_bar_own >= 30:
            power_prompt = "You can now use a powerful move. The names of the powerful moves are: Megafireball, Super attack 2."
        if super_bar_own >= 120 or super_bar_own == 0:
            power_prompt = "You can now only use very powerful moves. The names of the very powerful moves are: Super attack 3, Super attack 4"
        # Create the last action prompt
        last_action_prompt = ""
        if len(self.previous_actions.keys()) >= 0:
            act_own_list = self.previous_actions["agent_" + str(side)]
            act_opp_list = self.previous_actions["agent_" + str(abs(1 - side))]

            if len(act_own_list) == 0:
                act_own = 0
            else:
                act_own = act_own_list[-1]
            if len(act_opp_list) == 0:
                act_opp = 0
            else:
                act_opp = act_opp_list[-1]

            str_act_own = INDEX_TO_MOVE[act_own]
            str_act_opp = INDEX_TO_MOVE[act_opp]

            last_action_prompt += f"Your last action was {str_act_own}. The opponent's last action was {str_act_opp}."

        reward = self.reward

        # Create the score prompt
        score_prompt = ""
        if reward > 0:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif reward < 0:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )

        # Assemble everything
        context = f"""{position_prompt}
{power_prompt}
{last_action_prompt}
Your current score is {reward}. {score_prompt}
To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
"""

        return context

    def get_moves_from_llm_new(
        self,
    ):
        """
        Get a list of moves from the language model.
        """

        # Filter the moves that are not in the list of moves
        invalid_moves = []
        valid_moves = []

        # If we are in the test environment, we don't want to call the LLM
        if os.getenv("DISABLE_LLM", "False") == "True":
            # Choose a random int from the list of moves
            logger.debug("DISABLE_LLM is True, returning a random move")
            return [random.choice(list(MOVES.values()))]

        while len(valid_moves) == 0:
            llm_stream = self.call_llm()

            for move in self.parse_llm_stream(llm_stream):
                cleaned_move_name = move.strip().lower()
                if cleaned_move_name in META_INSTRUCTIONS_WITH_LOWER.keys():
                    if self.player_nb == 1:
                        print(
                            f"[red] Player {self.player_nb} move: {cleaned_move_name}"
                        )
                    elif self.player_nb == 2:
                        print(
                            f"[green] Player {self.player_nb} move: {cleaned_move_name}"
                        )
                    valid_moves.append(cleaned_move_name)
                    yield cleaned_move_name
                else:
                    logger.debug(f"Invalid completion: {move}")
                    logger.debug(f"Cleaned move name: {cleaned_move_name}")
                    invalid_moves.append(move)

            if len(invalid_moves) > 1:
                logger.warning(f"Many invalid moves: {invalid_moves}")

        logger.debug(f"Next moves: {valid_moves}")

    def get_moves_from_llm(
        self,
    ) -> List[str]:
        """
        Get a list of moves from the language model.
        """

        # Filter the moves that are not in the list of moves
        invalid_moves = []
        valid_moves = []

        # If we are in the test environment, we don't want to call the LLM
        if os.getenv("DISABLE_LLM", "False") == "True":
            # Choose a random int from the list of moves
            logger.debug("DISABLE_LLM is True, returning a random move")
            return [random.choice(list(MOVES.values()))]

        while len(valid_moves) == 0:
            llm_stream = self.call_llm()

            # adding support for streaming the response
            # this should make the players faster!

            llm_response = ""

            for r in llm_stream:
                print(r, end="")
                llm_response += r

                # The response is a bullet point list of moves. Use regex
                matches = re.findall(r"- ([\w ]+)", llm_response)
                moves = ["".join(match) for match in matches]
                invalid_moves = []
                valid_moves = []
                for move in moves:
                    cleaned_move_name = move.strip().lower()
                    if cleaned_move_name in META_INSTRUCTIONS_WITH_LOWER.keys():
                        if self.player_nb == 1:
                            print(
                                f"[red] Player {self.player_nb} move: {cleaned_move_name}"
                            )
                        elif self.player_nb == 2:
                            print(
                                f"[green] Player {self.player_nb} move: {cleaned_move_name}"
                            )
                        valid_moves.append(cleaned_move_name)
                    else:
                        logger.debug(f"Invalid completion: {move}")
                        logger.debug(f"Cleaned move name: {cleaned_move_name}")
                        invalid_moves.append(move)

                if len(invalid_moves) > 1:
                    logger.warning(f"Many invalid moves: {invalid_moves}")

            logger.debug(f"Next moves: {valid_moves}")
            return valid_moves

    def parse_llm_stream(self, llm_stream, stream: bool = False):
        llm_response = ""
        first = True
        start = time.time()
        if stream:
            for r in llm_stream:
                print(r, end="")

                # Calculate time to first token
                if first:
                    end = time.time()
                    ttft = end - start
                    self.ttft_list.append(ttft)
                    first = False

                llm_response += r
                # Use regex to find complete moves in real-time
                while '\n' in llm_response:
                    line, llm_response = llm_response.split('\n', 1)
                    match = re.match(r"- ([\w ]+)", line)
                    if match:
                        move = match.group(1).strip()
                        yield move
            # Handle any remaining text in the buffer
            if llm_response:
                match = re.match(r"- ([\w ]+)", llm_response)
                if match:
                    move = match.group(1).strip()
                    yield move
        else:
            start = time.time()
            for r in llm_stream:
                llm_response += r.delta

                # Calculate time to first token
                if first:
                    end = time.time()
                    ttft = end - start
                    self.ttft_list.append(ttft)
                    first = False

            # Process the accumulated response
            matches = re.findall(r"- ([\w ]+)", llm_response)
            moves = ["".join(match) for match in matches]
            invalid_moves = []
            valid_moves = []
            moves = []
            while '\n' in llm_response:
                line, llm_response = llm_response.split('\n', 1)
                match = re.match(r"- ([\w ]+)", line)
                if match:
                    move = match.group(1).strip()
                    moves.append(move)
            # Handle any remaining text in the buffer
            if llm_response:
                match = re.match(r"- ([\w ]+)", llm_response)
                if match:
                    move = match.group(1).strip()
                    moves.append(move)
            yield from moves

    def call_llm(
        self,
        temperature: float = 0.7,
        max_tokens: int = 50,
        top_p: float = 1.0,
    ):
        """
        Make an API call to the language model.

        Edit this method to change the behavior of the robot!
        """

        # Generate the prompts
        move_list = "- " + "\n - ".join([move for move in META_INSTRUCTIONS])
        system_prompt = f"""You are the best and most aggressive Street Fighter III 3rd strike player in the world.
Your character is {self.character}. Your goal is to beat the other opponent as quickly as possible. You respond with a bullet point list of moves.
{self.context_prompt()}
The moves you can use are:
{move_list}
----
Reply with a bullet point list of moves. The format should be: `- <name of the move>` separated by a new line.
Example if the opponent is close:
- Move closer
- Medium Punch

Example if the opponent is far:
- Fireball
- Move closer
Use the information provided to respond with the best next set of moves and always follow the format: `- <name of the move>` separated by a new line.\
"""

        user_prompt = "Your next moves are:"
        start_time = time.time()

        messages = [
            # ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt + "\n" + system_prompt),
        ]
        if self.model.startswith("samba"):
            print("USING SAMBA")
            model_name = self.model[6::]
            stream = self.chat_samba(user_prompt + "\n" + system_prompt, system_prompt, model_name)

            logger.debug(f"LLM call to {self.model}: {system_prompt}")
            logger.debug(f"LLM call to {self.model}: {time.time() - start_time}s")

            first = True
            start = time.time()
            for chunk in stream:
                end = time.time()
                if first:
                    self.ttft_list.append(end - start)
                    first = False

                yield chunk
            end = time.time()
            self.reponse_time_list.append(end - start)
        else:
            client = get_client(self.model)
            stream = client.stream_chat(messages)

            logger.debug(f"LLM call to {self.model}: {system_prompt}")
            logger.debug(f"LLM call to {self.model}: {time.time() - start_time}s")

            first = True
            start = time.time()
            for chunk in stream:
                end = time.time()
                # Calculate time to first token
                if first:
                    self.ttft_list.append(end - start)
                    first = False

                token = chunk
                yield chunk.delta

            end = time.time()
            self.reponse_time_list.append(end - start)
            # Calculate tokens per second in stream
            comp_tokens = token.raw['x_groq']['usage']['completion_tokens']
            comp_time = token.raw['x_groq']['usage']['completion_time']
            self.tps_list.append(comp_tokens/comp_time)

    def chat_samba(
            self,
            user_prompt: str,
            system_prompt: str = "You are a helpful assistant",
            model: str = "google/gemma-7b-it",
            api_key: str = os.getenv("SAMBA_API_KEY")
        ):
        url = 'https://sambaverse.sambanova.ai/api/predict'
        headers = {
            'Content-Type': 'application/json',
            'key': api_key or '973d1e73-c16f-47d8-8cb7-27f83eb2bf2b',
            'modelName': model
        }

        data = {
            "instance": json.dumps({
                "conversation_id": "sambaverse-conversation-id",
                "messages": [
                    {
                        "message_id": 0,
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "message_id": 1,
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }),
            "params": {
                "do_sample": {"type": "bool", "value": "true"},
                "max_tokens_to_generate": {"type": "int", "value": "1024"},
                "process_prompt": {"type": "bool", "value": "true"},
                "repetition_penalty": {"type": "float", "value": "1.0"},
                "return_token_count_only": {"type": "bool", "value": "false"},
                "select_expert": {"type": "str", "value": "gemma-7b-it"},
                "stop_sequences": {"type": "str", "value": ""},
                "temperature": {"type": "float", "value": "0.7"},
                "top_k": {"type": "int", "value": "50"},
                "top_p": {"type": "float", "value": "0.95"}
            }
        }

        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                json_response = json.loads(decoded_line)
                yield json_response['result']['responses'][0]['stream_token']
        tps = json_response['result']['responses'][0]['total_tokens_per_sec']
        self.tps_list.append(tps)