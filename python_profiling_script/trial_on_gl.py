import torch
import torch.nn as nn
import ipdb
import argparse 
import os 
import torch.multiprocessing as mp 

class DataParallelModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)


    def forward(self, x):
        x = self.block1(x)
        return x

def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    print(f"replicas:{replicas}")
	
    inputs = nn.parallel.scatter(input, device_ids)
    print(f"inputs:{type(inputs)}")
    for i in range(len(inputs)):
        print(f"input {i}:{inputs[i].shape}")
		
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    print(f"outputs:{type(outputs)}")
    for i in range(len(outputs)):
        print(f"output {i}:{outputs[i].shape}")
		
    result = nn.parallel.gather(outputs, output_device)
    return result

def train(): 
    model = DataParallelModel()
    x = torch.rand(16,10)
    result = data_parallel(model.cuda(),x.cuda(), [0,1])
    print(f"result:{type(result)}") 

parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
parser.add_argument('-n', '--nodes', default=1,
                    type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
args = parser.parse_args() 
args.world_size = args.gpus * args.nodes # world size now calculated by number of gpus and number of nodes 
    
os.environ['MASTER_ADDR'] = '169.254.3.1' 
os.environ['MASTER_PORT'] = '29509' 
os.environ['WORLD_SIZE'] = str(args.world_size) 
mp.spawn(train, nprocs = args.gpus, args = (args,)) 
