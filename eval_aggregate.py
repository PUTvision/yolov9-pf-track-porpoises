import click
import json
from glob import glob

def print_first_row(keys):
    # Print the first row of the table, with the keys as the column names
    
    namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
                'num_frames': 'Frames'
              }
    
    print('File Name | ', end='')
    for k in keys:
        print(f'{namemap[k]:8} | ', end='')
        
    # add a new line separator of -
    print()
    print('=' * (10 + len(keys) * 11))


def print_with_format(name, data):
    # Print as a table row, using | as separator and 20 characters for each column
    
    name = name.split('/')[-1].split('.')[0]
    
    print(f'{name:8}  | ', end='')
    for k, v in data.items():
        print(f'{float(v[list(v.keys())[0]]):8.2f} | ', end='')
        
    # add a new line separator of -
    print()
    print('-' * (10 + len(data.items()) * 11))
    


@click.command()
@click.option('--results', '-r', help='Path to result dir with jsons', required=True)
def main(results):
    jsons = sorted(glob(results + '/*.json'))
    
    json_0 = json.load(open(jsons[0], 'r'))
    
    total_results = {k: [] for k in json_0.keys()}
    
    print_first_row(json_0.keys())
    
    for json_file in jsons:
        json_data = json.load(open(json_file, 'r'))
        for k, v in json_data.items():
            total_results[k].append(v)
            
        print_with_format(json_file, json_data)
    
    aggregated_total_results = {}
    
    for k, v in total_results.items():
        aggregated_total_results[k] = {k: sum([x[k] for x in v]) / len(v) for k in v[0].keys()}
    
    print_with_format('Total', aggregated_total_results)
    
if __name__ == '__main__':
    main()
