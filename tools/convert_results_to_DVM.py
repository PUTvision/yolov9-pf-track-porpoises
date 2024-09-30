import numpy as np
import click
from pathlib import Path



class DetectionResults:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.results = {}
        self.uniqued_ids = {}
        
        print(f'[LOGS] Reading results from {file_name}')
        
        self.keypoints_processing = 'keypoints' in file_name
        print(f'[LOGS] Keypoints processing: {self.keypoints_processing}')
        
        self.keypoints_names = ['tongue', 'tail']
            
        
        if not self.keypoints_processing:
            with open(file_name, 'r') as f:
                for line in f.read().split('\n')[:-1]:
                    line = line.split(',')
                    
                    frame_id = int(line[0])
                    object_id = int(line[1])
                    
                    left_x = int(line[2])
                    top_y = int(line[3])
                    
                    width = int(line[4])
                    height = int(line[5])
                    
                    confidence = float(line[6])
                    
                    if frame_id not in self.results:
                        self.results[frame_id] = {}
                    
                    center_x = left_x + width / 2
                    center_y = top_y + height / 2
                    
                    self.results[frame_id][object_id] = (center_x, center_y, width, height, confidence)
                    
                    self.uniqued_ids[object_id] = None
        else:
            with open(file_name, 'r') as f:
                for line in f.read().split('\n')[:-1]:
                    line = line.split(',')
                    
                    frame_id = int(line[0])
                    object_id = int(line[1])
                    
                    keypoints = list(map(int, line[2:]))
                    
                    if frame_id not in self.results:
                        self.results[frame_id] = {}
                        
                    self.results[frame_id][object_id] = list(keypoints[i:i+2] for i in range(0, len(keypoints), 2))
                    
                    self.uniqued_ids[object_id] = None
        
        print(f'[LOGS] Results loaded')
        
    def convert_and_save(self, directory: str):
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.keypoints_processing:
            files_handles = {
                object_id: open(out_dir / f'{object_id}.csv', 'w') for object_id in self.uniqued_ids
            }
            
            for object_id in files_handles:
                files_handles[object_id].write('name,frame,image_x,image_y\n')
            
            
            for frame_id in sorted(list(self.results.keys())):
                objects = sorted(list(self.results[frame_id].keys()))
                
                for object_id in objects:
                    x, y, w, h, confidence = self.results[frame_id][object_id]

                    files_handles[object_id].write(f'porpoise_{object_id},{frame_id-1},{int(x)},{int(y)}\n')
        else:
            ids = {id: i+1 for i, id in enumerate(self.uniqued_ids)}
            
            files_handles = {
                object_id: open(out_dir / f'{ids[object_id]}_keypoints.csv', 'w') for object_id in self.uniqued_ids
            }
            
            for object_id in files_handles:
                files_handles[object_id].write('name,frame,image_x,image_y\n')
                
            for frame_id in sorted(list(self.results.keys())):
                objects = sorted(list(self.results[frame_id].keys()))
                
                for object_id in objects:
                    keypoints = self.results[frame_id][object_id]
                    
                    if np.any(np.array(keypoints) == -1):
                        continue
                    
                    for i, (x, y) in enumerate(keypoints):                        
                        files_handles[object_id].write(f'porp_{ids[object_id]}_{self.keypoints_names[i]},{frame_id-1},{int(x)},{int(y)}\n')
            
        
        for k, v in files_handles.items():
            v.close()
            
        print(f'[LOGS] Results saved in {out_dir}')
        

@click.command()
@click.option('--source', '-s', help='Source of the results', required=True)
@click.option('--output', '-o', help='Output directory', default='./dvm_results')
def main(source: str, output: str):
    if not Path(source).exists():
        raise FileNotFoundError(f"File {source} not found")
    
    if not Path(output).exists():
        Path(output).mkdir(parents=True, exist_ok=True)
        
    processor = DetectionResults(source)
    processor.convert_and_save(directory=output+"/"+processor.file_name.split("/")[-1].split(".")[0])


if __name__ == '__main__':
    main()
