from serpapi import GoogleSearch

params = {
  "engine": "google_flights",
  "departure_id": "CCU",
  "arrival_id": "GAU",
  "outbound_date": "2023-07-12",
#   "return_date": "2024-06-11",
  "currency": "INR",
  "hl": "en",
  "type": "2",
  "api_key": "8ff19b67f6340c26335822205ded1b109584575455d9af315afaecda9f567fe5"
}

search = GoogleSearch(params)
results = search.get_dict()
# print(results)
price_insights = results["price_insights"]
# print(price_insights)
price_history = price_insights["price_history"]
print(price_history)


# 61 days