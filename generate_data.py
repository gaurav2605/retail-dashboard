import pandas as pd
import random
from datetime import datetime, timedelta

items = ['Milk', 'Bread', 'Butter', 'Diapers', 'Beer', 'Eggs', 'Nutella', 'Coffee', 'Wine', 'Cheese']
data = []
start_date = datetime.now() - timedelta(days=30)

for i in range(1, 2500):
    # Generate a random date within the last 30 days
    date = start_date + timedelta(days=random.randint(0, 30))
    
    # INJECT ANOMALY: Massive spike in Beer & Diapers on weekends
    if date.weekday() >= 5: 
        basket = ['Beer', 'Diapers']
        if random.random() > 0.5: basket.append('Snacks')
    else:
        basket = random.sample(items, random.randint(1, 4))
        # Strong pairing logic
        if 'Bread' in basket and 'Butter' not in basket and random.random() > 0.3:
            basket.append('Butter')
        if 'Wine' in basket and 'Cheese' not in basket and random.random() > 0.2:
            basket.append('Cheese')
            
    data.append([f"T{1000+i}", date.strftime("%Y-%m-%d"), ",".join(basket)])

df = pd.DataFrame(data, columns=['Transaction_ID', 'Date', 'Items'])
df.to_csv('transactions_2000.csv', index=False)
print("Successfully generated transactions_2000.csv with", len(df), "rows!")
