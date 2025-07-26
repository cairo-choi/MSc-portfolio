import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Load GPT predictions ===
gpt_df = pd.read_csv("C:\\Users\\comer\\OneDrive - Dublin City University\\Bitcoin Thesis Files\\GPT_Code\\predictions\\gpt_test_predictions.csv")

# === Load true direction labels ===
eval_df = pd.read_csv("C:\\Users\\comer\\OneDrive - Dublin City University\\Bitcoin Thesis Files\\GPT_Code\\data\\reddit\\test\\true_test_labels.csv")

# === Fix the timestamp column and convert to standard date format ===
eval_df['date'] = pd.to_datetime(eval_df['timestamp'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

# === Optional: drop the original timestamp column ===
# eval_df.drop(columns=['timestamp'], inplace=True)

# === Keep only necessary columns ===
eval_df_clean = eval_df[['date', 'test_true_direction']]

# === Merge on date column ===
merged = pd.merge(gpt_df, eval_df_clean, on='date', how='inner')

print(f"Merged shape: {merged.shape}")
print(merged[['date', 'gpt_decision', 'test_true_direction']].head())

# === Encode predictions and true labels ===
y_pred = merged['gpt_decision'].map({'increase': 1, 'decrease': 0})
y_true = merged['test_true_direction'].astype(int)

# === Filter out any rows where prediction was NaN (e.g., unknown decision) ===
mask = ~y_pred.isna()
y_pred_clean = y_pred[mask]
y_true_clean = y_true[mask]

# === Compute evaluation metrics ===
accuracy = accuracy_score(y_true_clean, y_pred_clean)
precision = precision_score(y_true_clean, y_pred_clean, zero_division=0)
recall = recall_score(y_true_clean, y_pred_clean, zero_division=0)
f1 = f1_score(y_true_clean, y_pred_clean, zero_division=0)

# === Print results ===
print("\nEvaluation Results")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 score:  {f1:.2f}")
