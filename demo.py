from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from blur import blur, randomize
from hamming import str_to_bin, bin_to_str, encode, decode, syndrome, correct
from stego import insert, extract
import matplotlib.image as mpimg

# encode our original message using hamming(8,4)
original_message = raw_input("Enter Message to Send: ") 
original_message_bin = str_to_bin(original_message)
msg_len = len(original_message)
print("Input:")
print(original_message)

encoded_message = encode(original_message_bin)

# open up the image we want to use and get it ready to put data in
img = Image.open("image.bmp")
img = np.asarray(img)
image = mpimg.imread("image.bmp")
plt.title("Original Image")
plt.imshow(image)
plt.show()
# insert our message into the image, then blur the image to simulate
# compression/transmission/general data loss
img_with_message = insert(img, encoded_message)

print("Hiding Data in Image")
im = Image.fromarray(img_with_message)
im.save("image_steg.bmp")
steg_image = mpimg.imread("image_steg.bmp")
plt.title("Steg Image")
plt.imshow(steg_image)
plt.show()

blur_the_image = True # change this to not blur the image
if blur_the_image:
    blurred_img = randomize(img_with_message)
else:
    blurred_img = img_with_message

print("Blurring Image")
im = Image.fromarray(blurred_img)
im.save("image_blurred.bmp")
blurred_image = mpimg.imread("image_blurred.bmp")
plt.title("Blurred Image")
plt.imshow(blurred_image)
plt.show()
# extract the message (with errors) from the message, find out what the errors
# are thanks to hamming, and generate an error syndrome (basically an array
# that says "here's where the errors are")
extracted_msg = extract(blurred_img)
decoded = decode(extracted_msg)
decoded_str = bin_to_str(decoded)

print("")
print("Decoded string:")
print(decoded_str[:msg_len])
print("")

syndrome = syndrome(extracted_msg)

#print("")
#print("Syndrome:")
#print(syndrome.T[:100])
#print("")

# using the syndrome, correct the errors in the message, then decode the
# corrected version and convert it to a string
print("Correcting Errors:")
corrected = correct(extracted_msg, syndrome)
final_msg = decode(corrected)
final_msg_string = bin_to_str(final_msg)

# and we're done!
print("")
print("Output:")
print(final_msg_string[:msg_len])



