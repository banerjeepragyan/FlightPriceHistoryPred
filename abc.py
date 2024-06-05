import pickle

array = [1, 2, 3, 4, 5, 7]

name = "data/"+"abc"+"def"+".pkl"

with open(name, 'wb') as file:
    pickle.dump(array, file)

with open(name, 'rb') as file:
    loaded_array = pickle.load(file)
print(loaded_array)

