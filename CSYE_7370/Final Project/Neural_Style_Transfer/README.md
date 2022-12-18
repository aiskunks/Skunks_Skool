# Neural Style Transfer
![image](https://user-images.githubusercontent.com/91019023/208218641-0cd83cb1-975c-4e90-a144-b7e7a761192b.png)
### Introduction
Neural style transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation. Common uses for NST are the creation of artificial artwork from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs. Several notable mobile apps use NST techniques for this purpose, including Deep-Art and Prisma. This method has been used by artists and designers around the globe to develop new artwork based on existent style(s).
![image](https://user-images.githubusercontent.com/91019023/208218653-1cb75e09-4fa2-4a5a-891b-3880bb58ec0e.png)
### How to use
Download the VGG19 model (https://drive.google.com/file/d/1NRC_Ks9ydcA7nLttenNXOHtd2GNQWxzw/view?usp=sharing) and put it under the folder called pretrained-model, then run:
```bash
$ python demo.py
```
### Result
input image:
![image](https://user-images.githubusercontent.com/91019023/208218953-1938f33b-2cb9-4632-adf7-afcb65f0e118.png)

style image:
![image](https://user-images.githubusercontent.com/91019023/208218958-08e769e7-d985-43a6-8c43-577d839c8c6c.png)

output image:
![image](https://user-images.githubusercontent.com/91019023/208218982-99d9bd69-8785-4e38-9507-b664d4015076.png)
