import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Converts the csv output of VINS-Mono to TUM RGB-D format txt')
    parser.add_argument('input_file_path', help='Path to the csv file')
    parser.add_argument('output_file_path', help='Path to the output txt file')
    args = parser.parse_args()

    data = pd.read_csv(args.input_file_path, usecols=range(8), names=list('txyzwijk'))

    with open(args.output_file_path,'w') as f:
        for i in range(len(data['t'])):
            f.write(f"{data['t'][i]*1e-9} {data['x'][i]} {data['y'][i]} {data['z'][i]} {data['i'][i]} {data['j'][i]} {data['k'][i]} {data['w'][i]}\n")

if __name__ == '__main__':
    main()
