import pandas as pd

# =========================
# STEP 1: LOAD DATASET
# =========================
df = pd.read_csv("dataset.csv")

# =========================
# STEP 2: CREATE MOOD COLUMN
# =========================
def get_mood(valence, energy):
    if valence > 0.6 and energy > 0.6:
        return "Happy"
    elif valence < 0.4 and energy < 0.4:
        return "Sad"
    elif energy > 0.7:
        return "Energetic"
    else:
        return "Calm"

df['mood'] = df.apply(lambda x: get_mood(x['valence'], x['energy']), axis=1)

print("Mood Created:")
print(df[['valence','energy','mood']].head())


# =========================
# STEP 3: PREPROCESSING
# =========================

# ✅ IMPORTANT: Removed valence & energy to avoid data leakage
X = df[['danceability','loudness','tempo','acousticness']]
y = df['mood']

# Convert labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData Ready:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# =========================
# STEP 4: DECISION TREE (Independent Model)
# =========================
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_acc = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", dt_acc)

print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt))


# =========================
# STEP 5: BOOSTING (AdaBoost)
# =========================
from sklearn.ensemble import AdaBoostClassifier

# Weak learner
base_model = DecisionTreeClassifier(max_depth=3)

# AdaBoost model
boost_model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    learning_rate=0.5
)

# Train
boost_model.fit(X_train, y_train)

# Predict
y_pred_boost = boost_model.predict(X_test)

# Accuracy
boost_acc = accuracy_score(y_test, y_pred_boost)

print("\nBoosting (AdaBoost) Accuracy:", boost_acc)
import matplotlib.pyplot as plt

# Accuracy values
models = ['Decision Tree', 'AdaBoost']
accuracies = [dt_acc, boost_acc]

# Plot
plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, str(round(v,2)), ha='center')

plt.show()
print("\n🎵 Music Mood Prediction Demo")

# Example new song input
new_song = [[0.6, -5.0, 120, 0.3]]  
# format: [danceability, loudness, tempo, acousticness]

# Scale it
new_song_scaled = scaler.transform(new_song)

# Predict using Decision Tree
prediction = dt.predict(new_song_scaled)

# Convert back to label
predicted_mood = le.inverse_transform(prediction)

print("Predicted Mood:", predicted_mood[0])
import matplotlib.pyplot as plt

models = ['Decision Tree', 'AdaBoost']
accuracy = [dt_accuracy, boost_accuracy]

plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()