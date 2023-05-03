import torch
import torch.distributed as dist 
import argparse 
import os 
import torch.multiprocessing as mp 
import torch.distributed as dist 

def train(gpu, args): 
    '''
    # Initialize the distributed process group
    dist.init_process_group(backend='nccl', init_method='env://')
    ''' 
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
    # Get the rank and size of the process group
    '''
    rank = dist.get_rank()
    size = dist.get_size()
    ''' 
    size = args.world_size 
    print("rank: {} size: {}".format(rank, size)) 
    '''
    # Create a tensor with rank as its value
    tensor = torch.tensor([1, 1]).cuda(gpu) 
    print("rank: {}, tensor: {}".format(rank, tensor)) 
    ''' 
    '''
    # Create a list of tensors with different values
    tensors = torch.Tensor([float(gpu)] * size).cuda(gpu) 
    ''' 
    '''
    if rank == 0: 
        tensors = [torch.Tensor([float(i)] * size).cuda(gpu) for i in range(size)] 
    else: 
        tensors = None 
    ''' 
    tensors = torch.arange(4) + rank * 4 
    tensors = tensors.cuda(gpu) 
    '''
    if rank == 0: 
        output = [torch.empty(size).cuda(gpu) for _ in range(size)] 
    else: 
        output = None 
    ''' 
    output = torch.empty([size]).cuda(gpu) 
    # Perform the all-to-all communication
    dist.all_to_all_single(output, tensors) 
    '''
    dist.scatter(output, tensors if rank == 0 else None, src=0) 
    ''' 
    '''
    dist.gather(tensors, output, dst = 0) 
    ''' 
    '''
    # remember the sequence 
    if rank == 0: 
        # Print the output tensor
        print(f'Rank {rank}, output {output}') 
    ''' 
    print(f'Rank {rank}, output {output}') 

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
    '''
    os.environ['MASTER_ADDR'] = '10.157.244.233' 
    ''' 
    os.environ['MASTER_ADDR'] = '169.254.3.1' 
    os.environ['MASTER_PORT'] = '29509' 
    os.environ['WORLD_SIZE'] = str(args.world_size) 
    mp.spawn(train, nprocs = args.gpus, args = (args,)) 
