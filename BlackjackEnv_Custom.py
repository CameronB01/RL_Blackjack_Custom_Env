from stable_baselines3 import PPO
from BlackjackEnv import BlackEnv
from collections import Counter
import csv
from collections import defaultdict

env = BlackEnv()
env.reset()

models_dir = "models/PPO"
model_path = f"{models_dir}/4000000.zip"

model = PPO.load(model_path, env=env)

episodes = 5000000

wins = []
action_counts = {}
win_counts = defaultdict(int)
total_counts = defaultdict(int)



for ep in range(episodes):
    state, info = env.reset()
    terminated = False
    while not terminated:
        env.render()
        action, _ = model.predict(state)
        new_state, reward, terminated, truncated, info = env.step(action.item())
        if action == 1:
            print('Hit')
        else:
            print('Stand')
        print(state)
        print(reward)
        print(terminated)
        if terminated:
            wins.append(reward)


        key = (state['player_score'], state['dealer_card'])
        if key not in action_counts:
            action_counts[key] = {'Stand': 0, 'Hit': 0}
        if action == 1:
            action_counts[key]['Hit'] += 1
        elif action == 0:
            action_counts[key]['Stand'] += 1

        
        if terminated:
            # Increment win count and total count for the combination of player_score and dealer_card
            key = (state['player_score'], state['dealer_card'])
            total_counts[key] += 1
            if reward > 0:
                win_counts[key] += 1
        
        state = new_state



# Overall win loss percentages
count = Counter(wins)
for key, value in count.items():
    print(f"{key}: {value} -> ({value/episodes})")

# Hand info and preferred action
sorted_actions = sorted(action_counts.items(), key=lambda x: (x[0][0], x[0][1]))

for key, value in sorted_actions:
    win = win_counts[key]
    total = total_counts[key]
    win_percentage = win / total if total != 0 else 0

    preferred_action = 'Hit' if value['Hit'] > value['Stand'] else 'Stand' if value['Stand'] > value['Hit'] else 'Tie'
    print(f"Player Score: {key[0]}, Dealer Card: {key[1]}, Hit count: {value['Hit']}, Stand count: {value['Stand']}, Preferred Action: {preferred_action}, Wins: {win}, Win Percentage: {win_percentage}")



# Writing to a CSV
player_scores = sorted(set(key[0] for key in action_counts.keys()))
dealer_cards = sorted(set(key[1] for key in action_counts.keys()))

with open('preferred_actions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    writer.writerow(['Player Score\Dealer Card'] + dealer_cards)
    
    for score in player_scores:
        row_data = [score]
        for card in dealer_cards:
            value = action_counts.get((score, card), {'Hit': 0, 'Stand': 0})
            preferred_action = 'Hit' if value['Hit'] > value['Stand'] else 'Stand' if value['Stand'] > value['Hit'] else 'Tie'
            row_data.append(preferred_action)
        writer.writerow(row_data)

print("CSV file generated successfully!")


env.close() 


