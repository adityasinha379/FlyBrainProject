import numpy as np
import matplotlib.pyplot as plt

# Input: video - sequence of images: b/w
def create_input(angular_velocity,t, bar_width=11.5,  H=48, V=32, field = 320):
    ommatidia_field = field/H
    dt = t[1]-t[0]
    bar = np.zeros((H,len(t)))
    bar_edges = np.zeros((H,len(t)))
    for i in range(H):
        bar_edges[i,int(i*(ommatidia_field/angular_velocity)/dt*1000)] = 1
        bar_edges[i,int(i/dt*1000*ommatidia_field/angular_velocity+(bar_width/angular_velocity)/dt*1000)] = 1
        bar[i,int(i*(ommatidia_field/angular_velocity)/dt*1000):int(i/dt*1000*ommatidia_field/angular_velocity+(bar_width/angular_velocity)/dt*1000)] = 1
    return bar, bar_edges

def delay(x,tau,dt):
    output = np.zeros_like(x)
    for i in range(len(x)-1):
        output[i+1] = x[i]+output[i]-output[i]/tau*dt
    return output

def medulla(input,tau,dt):
    # input from pair of ommatidia to HR detector pair. Shape: (2,t)
    output = np.zeros((4,np.shape(input)[1]))
    #print(np.shape(input))
    output[0,:] = delay(input[0],tau['1'],dt)*delay(input[1],tau['b'],dt)
    output[1,:] = delay(input[0],tau['b'],dt)*delay(input[1],tau['1'],dt)
    output[2,:] = delay(input[0],tau['2'],dt)*delay(input[1],tau['b'],dt)
    output[3,:] = delay(input[0],tau['b'],dt)*delay(input[1],tau['2'],dt)
    
    return output  #(4,t)

def lobulla(input):
    # input shape: (l//2,4,t)
    sum_out = np.sum(input, axis=0)     # (4,t)
    lob_out = (sum_out[0]-sum_out[1])+(sum_out[2]-sum_out[3]+1e-8) # (1,t)
    return sum_out, lob_out

def AVDU(sum_out,lob_out,tau,dt):  # sum_out: (4,t), lob_out: (1,t)
    avdu_out = np.zeros((2,len(lob_out)))
    avdu_out[0] = sum_out[0]/(sum_out[2]+1e-8) - delay(lob_out,tau['s'],dt)       # 1e-8 for stability
    avdu_out[1] = sum_out[1]/(sum_out[3]+1e-8) + delay(lob_out,tau['s'],dt)
    avdu_out[0] = delay(avdu_out[0],tau['s'],dt)
    avdu_out[1] = delay(avdu_out[1],tau['s'],dt)
    
    return avdu_out

def process_input(angular_velocity = 1000):
    l = 48
    b = 32
    dt = 0.1
    t_end = 372/angular_velocity*1000
    t = np.arange(0,t_end,dt)
    tau={'1':5,'2':15,'b':1,'s':10}   # in ms
    
    input, input_edge = create_input(angular_velocity,t)
    med_out = np.zeros((l//2,4,len(t)))
    
    for j in range(l//2):
        med_out[j,:,:] = b*medulla(input_edge[2*j:2*j+2,:],tau,dt)


    sum_out, lob_out = lobulla(med_out)   # sum_out for posititional features (landmarks), lob_out for AVDU inputs (motion)
    avdu_out = AVDU(sum_out,lob_out,tau,dt)
    
    pos_out = np.zeros((16,len(t)))
    for i in range(0,l,l//16):
        pos_out[int(i/l*16),:] = np.sum(input[i:i+l//16,:],axis=0)
    return  t,pos_out, avdu_out