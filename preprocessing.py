import pandas as pd
import re

def preprocess_return_earned(return_earned):
    if pd.isna(return_earned):
        return None
    elif "Negative" in return_earned:
        return 0
    elif "More than" in return_earned:
        return int(re.findall(r'\d+', return_earned)[0]) + 1
    else:
        numbers = re.findall(r'\d+', return_earned)
        return (int(numbers[0]) + int(numbers[1])) // 2

def extract_numerical_value(text):
    pattern = r"\d+"
    matches = re.findall(pattern, text)
    if len(matches) == 1:
        return int(matches[0])
    elif len(matches) == 2:
        return (int(matches[0]) + int(matches[1])) // 2
    else:
        return None

df = pd.read_excel("datasets/dataset.xlsx")
column_names = df.columns.tolist()
df = df[df['Percentage of Investment'] != "Don't Want to Reveal"]
df = pd.get_dummies(df, columns=['City'],dtype=int)
df = pd.get_dummies(df,columns=['Gender'],dtype=int)
df = pd.get_dummies(df,columns=['Marital Status'],dtype=int)
df = pd.get_dummies(df,columns=['Age'],dtype=int)
df = pd.get_dummies(df,columns=['Education'],dtype=int)
df = pd.get_dummies(df,columns=['Role'],dtype=int)
df = pd.get_dummies(df,columns=['Investment Experience'],dtype=int)
df = pd.get_dummies(df,columns=['Investment Influencer'],dtype=int)
df = pd.get_dummies(df,columns=['Reason for Investment'],dtype=int)
df = pd.get_dummies(df,columns=['Risk Level'],dtype=int)
df = pd.get_dummies(df,columns=['Percentage of Investment'],dtype=int)
df = pd.get_dummies(df,columns=['Source of Awareness about Investment'],dtype=int)
df['Knowledge Level'] = df[['Knowledge level about different investment product','Knowledge level about sharemarket','Knowledge about Govt. Schemes']].mean(axis=1)
df['Knowledge Level'] = df['Knowledge Level'].astype(int)
df["Numeric Household Income"] = df["Household Income"].apply(extract_numerical_value)
df['Return Earned'] = df['Return Earned'].apply(preprocess_return_earned)

df = df.drop(columns='Household Income')
print(df)

df.to_excel("datasets/dataset1.xlsx", index=False)