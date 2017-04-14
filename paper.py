import numpy as np
from math import floor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def str_to_bin(string):
    """ take a string and return a list of integers (1 or 0) representing that
        string in ASCII """
    ret = list(string)
    # convert to binary representation
    ret = ['{:07b}'.format(ord(x)) for x in ret]
    # split the binary into
    ret = [[bit for bit in x] for x in ret]
    # flatten it and convert to integers
    ret = [int(bit) for sublist in ret for bit in sublist]
    return ret

def bin_to_str(binary):
    """ take a list of binary (integer ones and zeros) and return its ascii
        string representation """
    output = []
    for i in range(int(floor(len(binary)/7))):
        start = i * 7
        # this is gross
        char = binary[start:start+7]
        char = '0b' + ''.join([str(c) for c in char])
        output.append(int(char, 2))
    #print output
    st = ''.join([chr(o) for o in output if o != 0])
    return str(st)

def convert_msg(msg):
    while len(msg) % 3 != 0:
        # pad the message to length
        msg.append(0)
    bin_strings = [msg[3*i:3*i+3] for i in range(len(msg)/3)]
    return bin_strings

def convert_lsb(lsb):
    bin_strings = [lsb[5*i:5*i+5] for i in range(len(lsb)/5)]
    return bin_strings

def dec(binary):
    mult_arr = 2**np.arange(3)[::-1]
    ans = np.dot(binary,mult_arr)
    return int(ans)



msg = raw_input("Enter message to hide:")
msg = str_to_bin(msg)
print("")
print("Message in binary:")
print(msg)
print("")
raw_input()
print("Message grouped into 3 bits each (5,3)Hamming Code:")
print("")
msg = convert_msg(msg)
print(np.asarray(msg)[:10])
raw_input()

img = Image.open("image2.bmp")

img = np.asarray(img)
new = img.copy()

plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.title("Original Image")
plt.show()
print("")
print("Extracting LSBs from Original Image ...")
lsb = np.mod(img,2)
lsb_list = lsb.reshape(1,-1).tolist()[0]
raw_input()
print("Group into bits of 5:")
s = convert_lsb(lsb_list)
print(np.asarray(s)[:10])
raw_input()
print("")
H,W = img.shape

count = (H * (W))/5


if(len(msg)>len(s)):
    raise ValueError("Message too long to hide. Aborting.")
    

print("\nPadding message..")
while(len(s)!=len(msg)):
    msg.append([0,0,0])
len_msg_bits = len(msg)*3

a = np.asarray(msg)
s = np.asarray(s)

h =  np.mat('1,0,1,0,0;\
             0,1,0,1,0;\
             1,1,0,0,1')
raw_input()
print("H Matrix used:")
print(h)
print("")
lookup ={ 0 : [],
          1 : [4],
          2 : [3],
          4 : [2],
          3 : [1],
          5 : [0],
          6 : [0,1],
          7 : [1,2],
          }
raw_input()
print("")
print("Applying the algorithm specified in the paper ..")
for i in range(count):
    r = s[i]
    delta = a[i]
    
    synd = np.dot(r,h.transpose())
    indx = np.mod(synd + delta, 2)
    indx = dec(indx)
    #print(lookup[indx])
    
    # Flip bits
    for bit_idx in lookup[indx]:
        index = 5*i + bit_idx
        row = index / H
        col = index % W
        #print(i,lookup[indx], row, col)
        if(lsb[row,col]):
            new[row,col] -= 1
        else:
            new[row,col] += 1

raw_input()
new_img = Image.fromarray(new)
new_img.save("ham_image.bmp")

plt.imshow(new_img,cmap='gray',vmin=0,vmax=255)
plt.title("Final Image")
plt.show()

new_lsb = np.mod(new,2)
msg_bits = new_lsb.reshape(-1,5)
print("")
print("LSBs after running algorithm:")
print(msg_bits[:10])
print("")
raw_input()
deltas = np.mod(np.dot(msg_bits, h.transpose()),2)
print("Extracting message by multiplying with decode matrix:")
print(deltas[:10])

raw_input()
bin_msg_bits = deltas.reshape(1,-1).tolist()[0]
output_msg = bin_to_str(bin_msg_bits)
print("")
print("Output Message after extraction:")
print(output_msg)


newtemp = new.astype(np.int16)
oldtemp = img.astype(np.int16)
raw_input()
print("")
print("Running calculations")
total_pixels = H*W
change = newtemp - oldtemp
mean_sqr_error = np.sum((change)**2)/float(total_pixels)
PSNR = 10*np.log10(total_pixels/mean_sqr_error)
embedding_rate = float(len_msg_bits)/total_pixels
non_zero = np.count_nonzero(change)
change_density = float(non_zero)*100/ total_pixels
raw_input()
print("PSNR: ", PSNR)
print("Embedding Rate: ", embedding_rate)
print("Change Density: ", change_density,"%")