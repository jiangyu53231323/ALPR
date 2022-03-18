
import os

list = os.listdir("C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\lib")
for i in list:
    if i.endswith("453.lib"):
        print(i)
# print(list)