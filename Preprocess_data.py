import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv(r"C:\Users\saini\Downloads\model_training_dataset.csv")

# Limit the dataset to 1000 rows
df = df.sample(n=1000, random_state=42).reset_index(drop=True)

# Combine 'human_text' and 'ai_text' into one 'text' column and create 'label' column
human_texts = df['human_text'].dropna().tolist() # Extract non-null human texts
ai_texts = df['ai_text'].dropna().tolist() # Extract non-null AI texts

# Create new DataFrames for human and AI texts with labels (0 for human, 1 for AI)
human_df = pd.DataFrame({'text': human_texts, 'label': 0}) # Human-written
ai_df = pd.DataFrame({'text': ai_texts, 'label': 1}) # AI-generated

# Concatenate the human and AI text data into a single DataFrame
final_df = pd.concat([human_df, ai_df], ignore_index=True)

# Split the combined data into training and validation sets (80% training, 20% validation)
train_df, val_df = train_test_split(final_df, test_size=0.2)

# Save the processed datasets to new CSV files for later use
train_df.to_csv('train_dataset.csv', index=False)
val_df.to_csv('val_dataset.csv', index=False)

print("Preprocessing complete. Train and validation datasets created.")
