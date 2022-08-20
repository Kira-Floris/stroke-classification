import pandas as pd
import sqlite3

dataset = pd.read_csv('data/data.csv')

conn = sqlite3.connect('stroke_database')
c = conn.cursor()

c.execute('CREATE TABLE IF NOT EXISTS stroke (id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke)')
conn.commit()

dataset.to_sql('stroke', conn, if_exists='replace', index=False)

conn.close()