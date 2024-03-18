import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 從 CSV 檔案載入使用者樣本和餐廳資料
samples_df = pd.read_csv('samples.csv')
restaurants_df = pd.read_csv('pre_processed.csv')

# 將餐廳的營業狀態映射為數字值（0 表示已關閉，1 表示營業中）
restaurants_df['營業'] = restaurants_df['營業'].map({'Closed': 0, 'Open': 1})

# 將價格範圍切分為最小價格和最大價格
restaurants_df['最低價格'] = restaurants_df['價格'].apply(lambda x: int(x.strip('[,] ').split(',')[0]))
restaurants_df['最高價格'] = restaurants_df['價格'].apply(lambda x: int(x.strip('[,] ').split(',')[1]))

# 建立訓練數據集
train_data = []
for user_index, user_row in samples_df.iterrows():
    for restaurant_index, restaurant_row in restaurants_df.iterrows():
        sample = user_row.to_dict()
        sample.update({'Rating': restaurant_row['評分'], 'Open': restaurant_row['營業'], 'Price': restaurant_row['最低價格']})
        sample['選擇'] = 1 if user_row['User'] == f'User{restaurant_index+1}' else 0
        train_data.append(sample)

# 轉換為 DataFrame
train_df = pd.DataFrame(train_data)

# 準備特徵和目標變數
X = train_df.drop(['選擇', 'User'], axis=1)
y = train_df['選擇']


# 將數值特徵標準化或歸一化

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 針對性別特徵執行獨熱編碼，列名前綴為 'Gender'
X_train = pd.get_dummies(X_train, columns=['Gender'], prefix='Gender')
X_test = pd.get_dummies(X_test, columns=['Gender'], prefix='Gender')

# 使用 fillna() 方法填充缺失值
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
# 建立並擬合羅吉斯回歸模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 在測試集上預測概率
y_pred = model.predict(X_test)

# 輸出模型的準確率
print("模型準確率:", model.score(X_test, y_test))

# 定義新用戶的特徵
new_user = {'Age': 30, 'Gender': 'Female', 'Income': 100000}
new_user_data = []

for restaurant_index, restaurant_row in restaurants_df.iterrows():
    sample = new_user.copy()
    gender_male = 1 if new_user['Gender'] == 'Male' else 0
    gender_female = 1 if new_user['Gender'] == 'Female' else 0
    sample.update({'Rating': restaurant_row['評分'], 'Open': restaurant_row['營業'], 'Price': restaurant_row['最低價格'], 'Gender_Male': gender_male, 'Gender_Female': gender_female})
    new_user_data.append(sample)

# 建立包含正確特徵順序的新用戶 DataFrame
new_user_df = pd.DataFrame(data=new_user_data)


# 建立包含正確特徵順序的新用戶 DataFrame
new_user_df = pd.DataFrame(columns=X_train.columns, data=new_user_data)

# 使用默認值填充缺失值或在適當的地方刪除它們
new_user_df.fillna(0, inplace=True)

# 預測新用戶的餐廳選擇概率
restaurant_probabilities = model.predict_proba(new_user_df)[:, 1]

# 找到概率最高的餐廳
max_probability_index = restaurant_probabilities.argmax()
max_probability_restaurant = restaurants_df.loc[max_probability_index, '店名']
max_probability = restaurant_probabilities[max_probability_index]
# Output results
output_df = pd.DataFrame({'Restaurant': restaurants_df['店名'], 'Probability (%)': restaurant_probabilities * 100})
output_df['Probability (%)'] = output_df['Probability (%)'].apply(lambda x: f"{x:.2f}%")
output_df.to_csv('restaurant_probabilities.csv', index=False)

print(output_df)
print(f"\nRecommended restaurant: {max_probability_restaurant}, Probability: {max_probability * 100:.2f}%")