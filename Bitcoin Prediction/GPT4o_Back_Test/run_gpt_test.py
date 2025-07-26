import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime

# Load environment
load_dotenv("../../requirement files/api.env")
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Load data
reddit_df = pd.read_csv("../../data/reddit/test/reddit_input_no_dates.csv")
dates_df  = pd.read_csv("../../data/reddit/test/reddit_test_input.csv")

lstm_df = pd.read_csv("data/lstm_preds.csv")
gru_df  = pd.read_csv("data/gru_preds.csv")

# check matching rows
if len(reddit_df) != len(lstm_df) or len(gru_df) != len(lstm_df):
    raise ValueError("CSV files do not have the same number of rows!")

if len(dates_df) < len(reddit_df):
    raise ValueError("dates CSV does not have enough rows")

# Define LSTM and GRU metrics
lstm_metrics = {"accuracy": 0.66, "f1": 0.66, "precision": 0.76, "recall": 0.66}
gru_metrics  = {"accuracy": 0.70, "f1": 0.70, "precision": 0.70, "recall": 0.70}

results = []

for i in range(len(lstm_df)):
    reddit_text = reddit_df.iloc[i]['reddit_daily_text']
    lstm_pred = lstm_df.iloc[i]['lstm_pred_direction_test']
    gru_pred  = gru_df.iloc[i]['test_pred_direction']
    
    # Prompt for full GPT model
    prompt = f"""
You are an expert cryptocurrency analyst.

You must only answer using the data provided below. If you cannot answer from this data, respond with: {{"decision": "I don't know"}}.

Do not use any outside knowledge or assumptions beyond the provided data.

Your data sources are:

1️ Reddit sentiment headline text
"{reddit_text}"

2️ LSTM model prediction
• Direction: {"increase" if lstm_pred==1 else "decrease"}
• Accuracy: {lstm_metrics['accuracy']:.2f}
• F1: {lstm_metrics['f1']:.2f}
• Precision: {lstm_metrics['precision']:.2f}
• Recall: {lstm_metrics['recall']:.2f}

3️ GRU model prediction
• Direction: {"increase" if gru_pred==1 else "decrease"}
• Accuracy: {gru_metrics['accuracy']:.2f}
• F1: {gru_metrics['f1']:.2f}
• Precision: {gru_metrics['precision']:.2f}
• Recall: {gru_metrics['recall']:.2f}

Your task is to combine these sources and predict if the Bitcoin 30-day moving average will **increase or decrease**.  

Do not mention any dates in your answer.  

Return ONLY JSON:  
{{
  "decision": "increase" or "decrease"
}}
    """
    print(f" Processing row {i+1}/{len(lstm_df)}")
    print(f" Processing row {i+1}/{len(reddit_df)}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a cryptocurrency analyst."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=256,
        )
        answer_json = json.loads(response.choices[0].message.content)
        decision = answer_json.get("decision", "unknown")
    except Exception as e:
        print(f"Error on row {i}: {e}")
        decision = "error"
    
    results.append({
        "row_index": i,
        "reddit_text": reddit_text,
        "lstm": lstm_pred,
        "gru": gru_pred,
        "gpt_decision": decision
    })

# Save GPT results
out_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("results", exist_ok=True)
gpt_preds_path = f"results/gpt_predictions_{timestamp}.csv"
out_df.to_csv(gpt_preds_path, index=False)
print(f"GPT predictions saved to {gpt_preds_path}")

# ------------- attach dates from reddit_bitcoin_jan_2024.csv -------------
# check that dates column exists
if "date" not in dates_df.columns:
    raise ValueError("dates CSV must have a 'date' column")

if len(dates_df) != len(out_df):
    raise ValueError("dates file and GPT predictions do not match in rows")

# attach the dates to the GPT results
out_df["date"] = dates_df["date"]

# re-order columns
cols = ["date", "reddit_text", 
        #"lstm", "gru",
        "gpt_decision"]
final_df = out_df[cols]

final_path = f"gpt_predictions_only_reddit_{timestamp}.csv"
final_df.to_csv(final_path, index=False)
print(f"Final predictions WITH DATES saved to {final_path}")
