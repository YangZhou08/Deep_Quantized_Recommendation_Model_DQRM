import torch
import torch.nn as nn
import ipdb
import argparse 
import os 
import torch.multiprocessing as mp 
import torch.distributed as dist 

class DataParallelModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)


    def forward(self, x):
        x = self.block1(x)
        return x

def data_parallel(module, input, device_ids, output_device=None, gpu = None): 
    if not device_ids:
        return module(input)

    print("pass here") 
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

    i = torch.LongTensor([[0, 1, 2],
                        [3, 4, 5]])

    # Define the values of the non-zero elements
    v = torch.FloatTensor([1, 2, 3])

    # Create a sparse_coo tensor with dimension (10, 10)
    s = torch.sparse_coo_tensor(i, v, size=(10, 10)).cuda(gpu) 
    print(s.to_dense()) 
    s = s.to_dense().cuda(gpu) 
    s = dist.all_reduce(s, dist.ReduceOp.SUM) 
    '''
    print(s.to_dense()) 
    ''' 
    print(s) 
    return result 

def train(gpu, args): 
    rank = args.nr * args.gpus + gpu # make global rank 
    dist.init_process_group(
        backend = "nccl", 
        init_method = 'env://', 
        world_size = args.world_size, 
        rank = rank
    ) 
    torch.manual_seed(0) 
    torch.cuda.set_device(gpu) # TODO think about using cpu and change code 

    torch.set_printoptions(profile = "full") 

    print("********") 
    print("********") 
    model = DataParallelModel()
    x = torch.rand(16,10)
    result = data_parallel(model.cuda(0),x.cuda(0), [0,1], gpu) 
    print(f"result:{type(result)}") 

if __name__ == "__main__": 
    mp.freeze_support() 
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
        
    os.environ['MASTER_ADDR'] = '10.157.244.233' 
    os.environ['MASTER_PORT'] = '29509' 
    os.environ['WORLD_SIZE'] = str(args.world_size) 
    mp.spawn(train, nprocs = args.gpus, args = (args,)) 
