# BUG: Mixtral infinite tokens generated
from groq import Groq

client = Groq()

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

user_prompt = "Only output your next moves are:"
stream = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content":user_prompt,
        }
    ],
    model="mixtral-8x7b-32768",
    stream=True,
    temperature=0.9,
    max_tokens=None
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="", flush=True)


# import json

# json_payload = dict(
#     messages=[
#         {
#             "role": "system",
#             "content": system_prompt
#         },
#         {
#             "role": "user",
#             "content":user_prompt,
#         }
#     ],
#     model="mixtral-8x7b-32768",
#     stream=True,
#     temperature=0.1,
#     max_tokens=None
# )

# # with open('repro.json', 'w') as f:
#     # json.dump(json_payload, f, indent=4)