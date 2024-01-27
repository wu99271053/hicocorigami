import argparse
import pandas as pd 


parser = argparse.ArgumentParser(description='C.Origami Training Module.')




  # Data and Run Directories
parser.add_argument('--dir', default='2077',
                        type=str,
                        help='Random seed for training')

args = parser.parse_args()

file1 = 'combined_data.csv'

# Read the CSV files
df1 = pd.read_csv(f'{args.dir}/{file1}')
# Round the data to 2 decimal places
df1 = df1.round(2)


# Save the combined dataframe as a new CSV file
df1.to_csv(f'{args.dir}/combined_data.csv', index=False)