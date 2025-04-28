import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# Load data
df = pd.read_csv("gold_rate_history.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df[["Date", "Pure Gold (24 k)"]].dropna()
df = df.rename(columns={"Pure Gold (24 k)": "Price"})
df = df.sort_values("Date")

# Feature engineering
df["Days"] = (df["Date"] - df["Date"].min()).dt.days
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["RollingAvg"] = df["Price"].rolling(window=7, min_periods=1).mean()

# Features and target
X = df[["Days", "Month", "DayOfWeek", "RollingAvg"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
model.fit(X_train, y_train)

#Make predictions
pred=model.predict(X_test)
print(pred)

# Evaluate model performance
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print("Error:",mae)
print("Accuracy:",r2)

# GUI Prediction function
def predict_price():
    try:
        date_input = entry_date.get()
        future_date = datetime.strptime(date_input, "%Y-%m-%d")
        future_days = (future_date - df["Date"].min()).days
        month = future_date.month
        day_of_week = future_date.weekday()
        last_rolling_avg = df["RollingAvg"].iloc[-1]

        input_features = pd.DataFrame([[future_days, month, day_of_week, last_rolling_avg]],
                                      columns=["Days", "Month", "DayOfWeek", "RollingAvg"])
        prediction = model.predict(input_features)[0]

        messagebox.showinfo("Prediction", f"Predicted Gold Price on {date_input}: ₹{prediction:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Optional: Show model accuracy visually
def plot_predictions():
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color='teal', label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Gold Price: Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Gold Price Predictor")

tk.Label(root, text="Enter Future Date (YYYY-MM-DD):").pack(pady=5)
entry_date = tk.Entry(root)
entry_date.pack(pady=5)

tk.Button(root, text="Predict Price", command=predict_price).pack(pady=10)
tk.Button(root, text="Show Prediction Accuracy", command=plot_predictions).pack(pady=5)

root.mainloop()
