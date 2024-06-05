import numpy as np
from serpapi import GoogleSearch
from datetime import datetime
import pickle

airport_list = ["DEL", "BOM", "BLR", "MAA", "CCU", "HYD", "AMD", "GAU"]
current_date = datetime.now()
formatted_date = current_date.strftime('%Y-%m-%d')

data = []

for i in range(8):
    for j in range(8):
        if i == j:
            continue
        params = {
        "engine": "google_flights",
        "departure_id": airport_list[i],
        "arrival_id": airport_list[j],
        "outbound_date": formatted_date,
        "currency": "INR",
        "hl": "en",
        "type": "2",
        "api_key": "4ddc2ed2c3e52c83e1ccb052511e154ac18a7722b26e852e2ab446e31af1b6f4" #"8ff19b67f6340c26335822205ded1b109584575455d9af315afaecda9f567fe5"
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            price_insights = results["price_insights"]
            price_history = price_insights["price_history"]
            days = len(price_history)
            price_history_formatted = []
            for k in range(days):
                price_history_formatted.append(price_history[k][1])
            name = "data/"+airport_list[i]+airport_list[j]+".pkl"
            with open(name, 'wb') as file:
                pickle.dump(price_history_formatted, file)
        finally:
            print(airport_list[i], " -> ", airport_list[j])

print("DONE")