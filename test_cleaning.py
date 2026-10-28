
import pandas as pd

from analytics.cleaning import (
    get_data_quality_report,
    get_missing_value_report,
    clean_dataset,
)


sample_data = pd.DataFrame({
    "Customer": ["A", "B", "A", "C"],
    "City": [" Delhi ", None, " Delhi ", "Noida"],
    "Sales": [5000, 3000, 5000, None],
})

print("ORIGINAL DATA")
print(sample_data)

print("\nDATA QUALITY REPORT")
print(get_data_quality_report(sample_data))

print("\nMISSING VALUE REPORT")
print(get_missing_value_report(sample_data))

cleaned_df, cleaning_log = clean_dataset(sample_data)

print("\nCLEANED DATA")
print(cleaned_df)

print("\nCLEANING LOG")
for entry in cleaning_log:
    print("-", entry)

print("\nORIGINAL DATA AFTER CLEANING")
print(sample_data)
