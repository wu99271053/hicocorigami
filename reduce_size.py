import argparse
import pandas as pd 


parser = argparse.ArgumentParser(description='C.Origami Training Module.')




  # Data and Run Directories
parser.add_argument('--dir', default='2077',
                        type=str,
                        help='Random seed for training')

args = parser.parse_args()

file1 = 'targets.csv'
file2 = 'computed_outputs.csv'
file3 = 'untrain_outputs.csv'

# Read the CSV files
df1 = pd.read_csv(f'{args.dir}/{file1}')
df2 = pd.read_csv(f'{args.dir}/{file2}')
df3 = pd.read_csv(f'{args.dir}/{file3}')

# Round the data to 2 decimal places
df1 = df1.round(2)
df2 = df2.round(2)
df3 = df3.round(2)

# Combine the dataframes by column
combined_df = pd.concat([df1, df2, df3], axis=1)

# Save the combined dataframe as a new CSV file
combined_df.to_csv(f'{args.dir}/combined_data.csv', index=False)