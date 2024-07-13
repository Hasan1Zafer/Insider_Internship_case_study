import os
import pandas as pd

def load_data(file_path, columns):
    data = []
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                parts = line.strip().split(';')
                if len(parts) == len(columns):
                    data.append(parts)
                else:
                    print(f"Skipping line {line_number}: {line.strip()}")
            except Exception as e:
                print(f"Error processing line {line_number}: {e}")
    return pd.DataFrame(data, columns=columns)

# Load data
product_categories = load_data('data/Product_Categories.txt', ['product_id', 'category'])
product_explanation = load_data('data/Product_Explanation.txt', ['product_id', 'description'])

# text preprocessing
def preprocess_text(text):
    text = text.lower()
    # Remove punctuation
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

data = pd.merge(product_categories, product_explanation, on='product_id')
data.dropna(inplace=True)
data['processed_description'] = data['description'].apply(preprocess_text)

data.to_csv('data/processed_data.csv', index=False)
print("Data preprocessing completed and saved to 'data/processed_data.csv'.")
