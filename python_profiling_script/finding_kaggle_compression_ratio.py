n_l = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572] 
ccc = [128,  128, 128,      128,     128,  24, 128,  128, 3, 128,   128,  128,     128,  27, 128,   128,     10, 128,  128,  4, 128,     18, 15, 128,    105, 128] 

'''
n_l = [9980200, 26095, 17224, 7383, 20152, 3, 7112, 1435, 62, 9756762, 1332128, 314263, 10, 2208, 11168, 122, 4, 971, 14, 9994101, 7267918, 9946670, 415284, 12422, 102, 36] 
ccc = [2048,    2048,  2048,  2048, 2048,  3, 2048, 1435, 62, 2048,    2048,    2048,   10, 2048, 2048,  122, 4, 971, 14, 2048,    2048,    2048,    2048,   2048,  102, 36] 
''' 

ly_d = [13, 512, 256, 64, 16, 512, 256, 1] 

'''
ly_d = [13, 512, 256, 64, 512, 512, 256, 1] 
''' 

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

print("total message size: {}".format((lin_param + emb_param) * 4)) 

print("message size after specified sparsification: {}".format((lin_param + ccc_param) * 4)) 
'''
print("message size after specified sparsification and EMB 8-bit quantization: {}".format(lin_param * 4 + ccc_param)) 
''' 

byte_number = 0.5 
print("message size after specified sparsification and EMB 8-bit quantization: {}".format(lin_param * 4 + ccc_param * byte_number)) 

print("embedding sparsity: {}".format(float(ccc_param)/emb_param)) 

print("sparsification effect: {}".format(float(ccc_param + lin_param)/(emb_param + lin_param))) 

print("sparsification plus quantization effect: {}".format(float(ccc_param * 8 + lin_param * 32)/(emb_param * 32 + lin_param * 32))) 

emb_param = 0 
ccc_param = 0 

lin_param = 0 
'''
for item in n_l2:  
    emb_param += item * 64 
print("embedding parameters are {}".format(emb_param)) 

for item in ccc2: 
    ccc_param += item * 64 
print("gradient communication for parameters of number {}".format(ccc_param)) 

for i in range(1, len(ly_d2)): 
    lin_param += ly_d2[i] * ly_d2[i - 1] 
print("mlp parameters {}".format(lin_param)) 
''' 
