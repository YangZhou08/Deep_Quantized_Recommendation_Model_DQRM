n_l = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572] 
ccc = [128,  128, 128,      128,     128,  24, 128,  128, 3, 128,   128,  128,     128,  27, 128,   128,     10, 128,  128,  4, 128,     18, 15, 128,    105, 128] 

ly_d = [13, 512, 256, 64, 16, 512, 256, 1] 

emb_param = 0 
ccc_param = 0 

lin_param = 0 

for item in n_l: 
    emb_param += item * 16 
print("embedding parameters are {}".format(emb_param)) 

for item in ccc: 
    ccc_param += item * 16 
print("gradient communication for parameters of number {}".format(ccc_param)) 

for i in range(1, len(ly_d)): 
    lin_param += ly_d[i] * ly_d[i - 1] 
print("mlp parameters {}".format(lin_param)) 

print("embedding sparsity: {}".format(float(ccc_param)/emb_param)) 

print("sparsification effect: {}".format(float(ccc_param + lin_param)/(emb_param + lin_param))) 

print("sparsification plus quantization effect: {}".format(float(ccc_param * 8 + lin_param * 32)/(emb_param * 32 + lin_param * 32))) 


