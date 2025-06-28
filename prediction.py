import joblib
import pandas as pd

# Load model
loaded_model = joblib.load('player_behaviour_model.pkl')

# Data Player
new_player_data = pd.DataFrame([{
    'blocks_mined': 50000,
    'crops_harvested': 2000,
    'logs_broken': 300,
    'fish_caught': 50,
    'mobs_killed': 8000,
    'blocks_placed': 1000,
    'items_enchanted': 5,
    'items_crafted': 20,
    'potions_brewed': 10,
    'animals_bred': 10
}])

predicted_label = loaded_model.predict(new_player_data)
print("Hasil: ", predicted_label[0])
