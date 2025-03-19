import pandas as pd

# Đọc file CSV gốc
df = pd.read_csv('data/mock/test.csv')

# Tách thành hai DataFrame riêng biệt cho tiếng Anh và tiếng Việt
df_en = df[['en']]
df_vi = df[['vi']]

# Lưu thành hai file CSV riêng biệt
df_en.to_csv('test.en', index=False)
df_vi.to_csv('test.vi', index=False)

print("Đã tách file thành công!")
print("File tiếng Anh: english.csv")
print("File tiếng Việt: vietnamese.csv") 