import pickle

with open("custom_data.pkl", "rb") as f:
    data = pickle.load(f)

new_X = []
new_y = []

for x, y in zip(data['X'], data['y']):
    if y != "Daftar":
        new_X.append(x)
        new_y.append(y)

with open("custom_data.pkl", "wb") as f:
    pickle.dump({'X': new_X, 'y': new_y}, f)

print("Muvaffaqiyatli o'chirildi!")