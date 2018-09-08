#import numpy as np
import random
n = 5
d = 10
z = 10
print(str(n))
print(str(d))
print(str(z))
for i in range(1,n+1):
	num = random.randint(1,5)
	
	for j in range(num):
		word_id = random.randint(1,d)
		freq = random.randint(1, 100)
		print(str(i) + " " + str(word_id) + " " + str(freq))


