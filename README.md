# UNO card recogntion game using Machine Learning
#### UNO game cards are recognized(color and number of the card) with the use of a webcam using Random Forest classification which is a Machine Learning technique.
The below images are sample photos of the datatset being used

<p align="center">
  <img src="Sample dataset/1.jpg" width="350" title="hover text">
  <img src="Sample dataset/6.jpg" width="350" alt="accessibility text">
</p>

### Dataset creation
1) First step is to create the dataset, which was created using a standard camera.<br>
2) Dataset is created 100 images of each number of cards, image processing such as
augmentation and RGBtoGRAY is implemented.<br>
3) For classification Random Forest is used and for feature extraction LBP is used. Local Binary Pattern (LBP) is a simple yet very efficient texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number. <br>
### Main Code
1)  There are three sections in the code (Main Code.py)<br>
a. Training Dataset<br>
b. Testing with Camera<br>
c. Testing with file load<br>
2) Significant results were produced both testing with the loaded file and with live stream.

### **Run the Main code.py and providing the address of the dataset. 

