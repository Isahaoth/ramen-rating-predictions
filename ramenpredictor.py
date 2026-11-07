import pandas as pd

df = pd.read_csv("ramen-ratings.csv")
df= df[df.Stars !="Unrated"]

df["is_spicy"] = df["Variety"].str.contains("spicy", case=False).astype(int)
df["is_chicken"] = df["Variety"].str.contains("chicken", case=False).astype(int)
df["is_creamy"] = df["Variety"].str.contains("creamy", case=False).astype(int)
df["is_tomato"] = df["Variety"].str.contains("tomato", case=False).astype(int)
df["is_shrimp"] = df["Variety"].str.contains("shrimp", case=False).astype(int)
df["is_beef"] = df["Variety"].str.contains("beef", case=False).astype(int)

df = df.drop(columns=["Variety"])

x = df[["Brand", "Style", "Country", "is_creamy", "is_spicy", "is_chicken", "is_tomato", "is_shrimp", "is_beef"]]
y = df["Stars"]

x.join(y).head(3)

x = pd.get_dummies(x, columns=["Brand", "Style", "Country"], dtype=int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

from sklearn.metrics import mean_absolute_error
prediction = model.predict(x_test)
mae = mean_absolute_error(y_test, prediction)

newRamen = pd.DataFrame({
    "Brand": ["Amino", "Samyang Foods"],
    "Style": ["Pack", "Pack"],
    "Country": ["Poland", "South Korea"],
    "is_spicy": [0, 1],
    "is_chicken": [0, 0],
    "is_creamy": [1, 1],
    "is_tomato": [1, 0],
    "is_shrimp": [0, 0],
    "is_beef": [0, 1]
})

newRamen = pd.get_dummies(newRamen, columns=["Brand", "Style", "Country"], dtype=int)
newRamen = newRamen.reindex(columns=x.columns, fill_value=0)
stars = model.predict(newRamen)
print("Predicted amount of stars Amino soup:", round(stars[0], 2))
print("Predicted amount of stars Samyang Foods soup:", round(stars[1], 2))
print("mae:", round(mae, 2))