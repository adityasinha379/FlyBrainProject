import numpy as np
import matplotlib.pyplot as plt

# Input: video - sequence of images: b/w
def create_input(l=32,b=48,t=48):
    # creates input of shape (l,b,t)
    input = np.zeros((l,b,t),dtype=np.bool)
    for i in range(t):
        input[:,(i*l)//t,i]=True
    return input

def delay(x,tau):
    dt = 0.1
    output = np.zeros_like(x)
    for i in range(len(x)-1):
        output[i+1] = x[i]+output[i]-output[i]/tau*dt
    return output

def medulla(input,tau):
    # input from pair of ommatidia to HR detector pair. Shape: (2,t)
    output = np.zeros((4,np.shape(input)[1]))
    
    output[0,:] = delay(input[0],tau['1'])*delay(input[1],tau['b'])
    output[1,:] = delay(input[0],tau['b'])*delay(input[1],tau['1'])
    output[2,:] = delay(input[0],tau['2'])*delay(input[1],tau['b'])
    output[3,:] = delay(input[0],tau['b'])*delay(input[1],tau['2'])
    
    return output  #(4,t)

def lobulla(input):
    # input shape: (l,b//2,4,t)
    sum_out = np.sum(np.sum(input, axis=0),axis=0)     # (4,t)
    lob_out = (sum_out[0]-sum_out[1])/(sum_out[2]-sum_out[3]+1e-8) # (1,t)
    return sum_out, lob_out

def AVDU(sum_out,lob_out,tau):  # sum_out: (4,t), lob_out: (1,t)
    avdu_out = np.zeros((2,len(lob_out)))
    avdu_out[0] = sum_out[0]/(sum_out[2]+1e-8) - delay(lob_out,tau['s'])       # 1e-8 for stability
    avdu_out[1] = sum_out[1]/(sum_out[3]+1e-8) + delay(lob_out,tau['s'])
    avdu_out[0] = delay(avdu_out[0],tau['s'])
    avdu_out[1] = delay(avdu_out[1],tau['s'])
    
    return avdu_out

def process_input():
    l = 32
    b = 48
    t = 50
    tau={'1':5,'2':15,'b':1,'s':10}   # in ms
    
    input = create_input(l,b,t)
    med_out = np.zeros((l,b//2,4,t))
    
    for i in range(l):
        for j in range(b//2):
            med_out[i,j,:,:] = medulla(input[i,2*j:2*j+2,:],tau)
    
    sum_out, lob_out = lobulla(med_out)   # sum_out for posititional features (landmarks), lob_out for AVDU inputs (motion)
    avdu_out = AVDU(sum_out,lob_out,tau)
    
    pos_out = np.zeros((16,t))
    for i in range(0,b//16,b):
        pos_out[i] = np.sum(input[:,i:i+b//16])
    return  pos_out, avdu_out