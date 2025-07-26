import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime
import re
import numpy as np

# Load environment
load_dotenv("api.env")
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Load Reddit data and convert date to Timestamp
dates_df = pd.read_csv(
    r"C:\Users\comer\OneDrive - Dublin City University\Bitcoin Thesis Files\GPT_Code\data\reddit_train\reddit_input_train.csv"
)
dates_df['date'] = pd.to_datetime(dates_df['date'])

# Define missing dates
missing_dates = pd.to_datetime([
    "2018-10-17", "2018-10-18", "2018-10-19", "2018-10-20", "2018-10-21", "2018-10-22", "2018-10-23", "2018-10-24",
    "2018-10-25", "2018-10-26", "2018-10-27", "2018-10-28", "2018-10-29", "2018-10-30", "2018-10-31",
    "2018-12-12", "2018-12-13", "2018-12-14", "2018-12-15", "2018-12-16", "2018-12-17", "2018-12-18", "2018-12-19",
    "2018-12-20", "2018-12-21", "2018-12-22", "2018-12-23", "2018-12-24", "2018-12-25", "2018-12-26", "2018-12-27",
    "2018-12-28", "2018-12-29", "2018-12-30", "2018-12-31", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15",
    "2019-02-16", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23",
    "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-02", "2019-03-03", "2019-03-04",
    "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12",
    "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20",
    "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28",
    "2019-03-29", "2019-03-30", "2019-03-31", "2019-12-29", "2019-12-30", "2019-12-31", "2020-12-05", "2020-12-06",
    "2020-12-07", "2020-12-08", "2020-12-09", "2020-12-10", "2020-12-11", "2020-12-12", "2020-12-13", "2020-12-14",
    "2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18", "2020-12-19", "2020-12-20", "2020-12-21", "2020-12-22",
    "2020-12-23", "2020-12-24", "2020-12-25", "2020-12-26", "2020-12-27", "2020-12-28", "2020-12-29", "2020-12-30",
    "2020-12-31", "2021-07-05", "2021-07-06", "2021-07-07", "2021-07-08", "2021-07-09", "2021-07-10", "2021-07-11",
    "2021-07-12", "2021-07-13", "2021-07-14", "2021-07-15", "2021-07-16", "2021-07-17", "2021-07-18", "2021-07-19",
    "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24", "2021-07-25", "2021-07-26", "2021-07-27",
    "2021-07-28", "2021-07-29", "2021-07-30", "2021-07-31"
])

# Use explicit placeholder for missing Reddit data
placeholder_data = pd.DataFrame({
    'date': missing_dates,
    'title': 'REDDIT_DATA_UNAVAILABLE'
})

# Combine and sort
dates_df_combined = pd.concat([dates_df, placeholder_data], ignore_index=True)
dates_df1 = dates_df_combined.sort_values(by='date').reset_index(drop=True)

# Store text and dates separately
reddit_df = dates_df1[['title']].rename(columns={'title': 'reddit_text'})
date_only_df = dates_df1[['date']]

# Load predictions, skipping headers manually
lstm_df = pd.read_csv(
    r"C:\Users\comer\OneDrive - Dublin City University\Bitcoin Thesis Files\GPT_Code\data\reddit_train\lstm_train_direction.csv",
    skiprows=1, names=["date", "lstm_pred_direction_all"]
)
gru_df = pd.read_csv(
    r"C:\Users\comer\OneDrive - Dublin City University\Bitcoin Thesis Files\GPT_Code\data\reddit_train\gru_train_direction.csv",
    skiprows=1, names=["date", "pred_direction_all"]
)

# Ensure row count matches
if not (len(reddit_df) == len(lstm_df) == len(gru_df)):
    raise ValueError("Mismatch in number of rows between Reddit, LSTM, and GRU files.")

# Model metric definitions (from training period)
lstm_metrics = {"accuracy": 0.66, "f1": 0.66, "precision": 0.76, "recall": 0.66}
gru_metrics = {"accuracy": 0.70, "f1": 0.70, "precision": 0.70, "recall": 0.70}

def clean_and_enhance_reddit_text(text):
    """Enhanced Reddit text processing for better sentiment signal"""
    if pd.isna(text) or text == '' or text == 'REDDIT_DATA_UNAVAILABLE':
        return 'REDDIT_DATA_UNAVAILABLE'
    
    # Convert to string and clean
    text = str(text).strip()
    
    # If still empty after cleaning
    if len(text) == 0:
        return 'REDDIT_DATA_UNAVAILABLE'
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Truncate intelligently - try to keep complete sentences
    if len(text) > 1200:  # Reduced from 1500 to leave room for prompt
        # Try to find last complete sentence within limit
        truncated = text[:1200]
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        if last_sentence_end > 800:  # Only if we have substantial content
            text = truncated[:last_sentence_end + 1]
        else:
            text = truncated + "..."
    
    return text

def format_reddit_data_for_prompt(reddit_text):
    """Format Reddit data for the prompt with clear status"""
    if reddit_text == 'REDDIT_DATA_UNAVAILABLE':
        return "STATUS: No Reddit sentiment data available for this date"
    else:
        # Count words for quality assessment
        word_count = len(reddit_text.split())
        if word_count < 10:
            return f"STATUS: Limited Reddit data available ({word_count} words)\nTEXT: {reddit_text}"
        else:
            return f"STATUS: Full Reddit sentiment data available ({word_count} words)\nTEXT: {reddit_text}"

# Enhanced prompt with few-shot examples and clear decision framework
def create_enhanced_prompt(reddit_text, lstm_pred, gru_pred):
    formatted_reddit = format_reddit_data_for_prompt(reddit_text)
    
    prompt = f"""You are a Bitcoin price direction prediction system. Follow these rules EXACTLY:

=== DECISION FRAMEWORK ===
1. If LSTM and GRU models AGREE: Follow their consensus regardless of Reddit data
2. If LSTM and GRU models DISAGREE and Reddit data unavailable: Follow the GRU model (higher F1 score: 0.70 vs 0.66)
3. If LSTM and GRU models DISAGREE and Reddit data available: Use Reddit sentiment as tiebreaker
4. Base decisions ONLY on provided data - no external knowledge

=== TRAINING EXAMPLES ===
Example 1:
LSTM: decrease, GRU: decrease, Reddit: No data
→ Decision: decrease (models agree)

Example 2:
LSTM: increase, GRU: decrease, Reddit: No data
→ Decision: decrease (GRU has higher F1: 0.70 vs 0.66)

Example 3:
LSTM: increase, GRU: decrease, Reddit: "bitcoin bullish momentum building"
→ Decision: increase (positive sentiment breaks tie)

=== YOUR DATA ===
 LSTM MODEL PREDICTION:
Direction: {"increase" if lstm_pred==1 else "decrease"}
Performance Metrics: Accuracy={lstm_metrics['accuracy']:.2f}, F1={lstm_metrics['f1']:.2f}, Precision={lstm_metrics['precision']:.2f}, Recall={lstm_metrics['recall']:.2f}

 GRU MODEL PREDICTION:
Direction: {"increase" if gru_pred==1 else "decrease"}
Performance Metrics: Accuracy={gru_metrics['accuracy']:.2f}, F1={gru_metrics['f1']:.2f}, Precision={gru_metrics['precision']:.2f}, Recall={gru_metrics['recall']:.2f}

 REDDIT SENTIMENT DATA:
{formatted_reddit}

The LSTM and GRU predictions are for the trend of Bitcoin's 30-day moving average.

=== ANALYSIS REQUIRED ===
1. Do the models agree? {"YES" if lstm_pred == gru_pred else "NO"}
2. Is Reddit data available? {"NO" if reddit_text == 'REDDIT_DATA_UNAVAILABLE' else "YES"}
3. Apply decision framework above

=== OUTPUT FORMAT ===
Return ONLY this JSON structure:
{{
    "models_agree": {"true" if lstm_pred == gru_pred else "false"},
    "reddit_available": {"false" if reddit_text == 'REDDIT_DATA_UNAVAILABLE' else "true"},
    "decision_rule_used": "[state which rule from framework was applied]",
    "decision": "[increase/decrease]",
    "confidence": "[high/medium/low]"
}}

Predict Bitcoin 30-day moving average direction:"""
    
    return prompt

# Process rows with enhanced GPT logic
results = []
for i in range(len(lstm_df)):
    reddit_text = clean_and_enhance_reddit_text(reddit_df.iloc[i]['reddit_text'])
    lstm_pred = lstm_df.iloc[i]['lstm_pred_direction_all']
    gru_pred = gru_df.iloc[i]['pred_direction_all']

    prompt = create_enhanced_prompt(reddit_text, lstm_pred, gru_pred)

    print(f"⏳ Processing row {i+1}/{len(lstm_df)}")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.05,  # Slightly increased for better reasoning variety
            messages=[
                {"role": "system", "content": "You are a systematic Bitcoin price direction prediction system that follows explicit rules."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=512,  # Increased for detailed response
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        answer_json = json.loads(response.choices[0].message.content)
        decision = answer_json.get("decision", "error")
        confidence = answer_json.get("confidence", "unknown")
        rule_used = answer_json.get("decision_rule_used", "unknown")
        
        # Validate decision format
        if decision not in ["increase", "decrease"]:
            print(f" Invalid decision '{decision}' on row {i}, defaulting to error")
            decision = "error"
            
    except Exception as e:
        print(f" Error on row {i}: {e}")
        decision = "error"
        confidence = "unknown"
        rule_used = "error"

    results.append({
        "row_index": i,
        "reddit_text": reddit_text,
        "reddit_status": "available" if reddit_text != 'REDDIT_DATA_UNAVAILABLE' else "unavailable",
        "lstm": lstm_pred,
        "gru": gru_pred,
        "models_agree": lstm_pred == gru_pred,
        "gpt_decision": decision,
        "confidence": confidence,
        "rule_used": rule_used
    })

    # Progress indicator every 50 rows
    if (i + 1) % 50 == 0:
        print(f" Completed {i+1}/{len(lstm_df)} rows")

# Save results with enhanced metadata
out_df = pd.DataFrame(results)
out_df["date"] = date_only_df["date"]

# Create analysis summary
total_rows = len(out_df)
reddit_available = sum(out_df['reddit_status'] == 'available')
models_agree = sum(out_df['models_agree'])
errors = sum(out_df['gpt_decision'] == 'error')

print(f"\n=== PROCESSING SUMMARY ===")
print(f"Total rows processed: {total_rows}")
print(f"Reddit data available: {reddit_available} ({reddit_available/total_rows*100:.1f}%)")
print(f"Models agree: {models_agree} ({models_agree/total_rows*100:.1f}%)")
print(f"Errors: {errors} ({errors/total_rows*100:.1f}%)")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("results", exist_ok=True)

# Full results with metadata
out_df.to_csv(f"results/gpt_enhanced_predictions_{timestamp}.csv", index=False)

# Clean results for analysis
final_df = out_df[["date", "reddit_text", "reddit_status", "lstm", "gru", "models_agree", "gpt_decision", "confidence", "rule_used"]]
final_df.to_csv(f"results/gpt_train_enhanced_predictions_{timestamp}.csv", index=False)

print(f" Enhanced predictions saved to results/gpt_train_enhanced_predictions_{timestamp}.csv")
print(f" Full metadata saved to results/gpt_enhanced_predictions_{timestamp}.csv")

# Quick validation check
if errors == 0:
    print(" No errors detected - all predictions generated successfully!")
else:
    print(f" {errors} errors detected - check the detailed output file for issues")