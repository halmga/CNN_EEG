# CNN for EEG Classification
This project was aimed at creating a neural network to classify different types sleep by using EEG signals. Our main model is explained in the [pdf](CNN_EEG_GroupPaper.pdf) and is a modification of a neural net proposed by Lawhern (2018). Aside from the models that are explained in our paper, we also tried a simple residual neural network and a modification of VGGnet16 (Simonyan & Zisserman, 2014). The initial thought was that VGGnet16 would perform well on our data since EEG signals can be represented as a spectrogram which is essentially an image which is what VGGnet16 was trained for. However, through training and testing of all models, we found that less deep networks with fewer layers performed the best on our data.

### Citations
Lawhern, V. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;interfaces. Journal of neural engineering, 15(5), 056013.<br><br>
Simonyan, K. & Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arXiv:1409.1556
