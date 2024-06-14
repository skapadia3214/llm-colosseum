import requests
import json
import os
from time import time

def chat_samba(
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant",
        model: str = "google/gemma-7b-it",
        api_key: str = os.getenv("SAMBA_API_KEY")
    ):
    url = 'https://sambaverse.sambanova.ai/api/predict'
    headers = {
        'Content-Type': 'application/json',
        'key': api_key,
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

    print("\n")
    print(f"{json_response['result']['responses'][0]=}")
    # print(f"Tokens generated: {len(json_response['result']['responses'][0]['tokens'])}")
    # print(f"Time Taken: {json_response['result']['responses'][0]['total_latency']}")
    # print(f"Tokens/sec: {json_response['result']['responses'][0]['total_tokens_per_sec']}")


system_prompt = """\
You are the best and most aggressive Street Fighter III 3rd strike player in the world.
Your character is Ken. Your goal is to beat the other opponent as quickly as possible. You respond with a bullet point list of moves.
You are very far from the opponent. Move closer to the opponent.Your opponent is on the left.

Your last action was High Punch. The opponent's last action was Medium Punch.
Your current score is -34.0. You are losing. Continue to attack the opponent but don't get hit.
To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.

The moves you can use are:
- Move Closer
 - Move Away
 - Fireball
 - Megapunch
 - Hurricane
 - Megafireball
 - Super attack 2
 - Super attack 3
 - Super attack 4
 - Low Punch
 - Medium Punch
 - High Punch
 - Low Kick
 - Medium Kick
 - High Kick
 - Low Punch+Low Kick
 - Medium Punch+Medium Kick
 - High Punch+High Kick
 - Jump Closer
 - Jump Away
----
ONLY reply with a bullet point list of moves. The format should be: `- <name of the move>` separated by a new line.
Example if the opponent is close:
- Move closer
- Medium Punch

Example if the opponent is far:
- Fireball
- Move closer

Use your knowledge about Street Fighter III 3rd strike player and the information given to you to come up with the best, most aggresive moves to beat your opponent as quickly as possible.
"""

user_prompt = "your next moves are:"
user_prompt = user_prompt

start = time()
count = 0
for chunk in chat_samba(user_prompt=user_prompt, system_prompt=system_prompt):
    print(chunk, end="")
    count += 1
end = time()

# print(f"\n(Cal) Tokens generated: {count}")
print(f"(Cal) Total response time: {end - start}")
# print(f"(Cal) Tokens/s: {count/(end - start)}")