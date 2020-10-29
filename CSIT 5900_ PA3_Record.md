# Report of Cifar10_CNN in Pytorch #
# CSIT 5900_Machine Learning_PA3 #
# SONG GE_20716021#

## Q1: Vary the number of hidden layers##
### Idea A: Vary the number of full connection layers ###
    '''
    ### decrease the number of fc ###
    '''
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        #x = self.fc2(x)
        return x
    '''
### Idea B: Vary the number of conv layers ###
    '''
    ### Increase the number of conv, when I just change the conv number, there is a error about the out_size and input_size, that's because of pooling layer compress the image to 1*1 matrix, so we can delete(not suitable here) or use "AdaptiveMaxPool" to instead the pooling layer, which can force the size of data to be 5*7. ###
    '''
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adapt = nn.AdaptiveMaxPool2d((5,7))
        self.conv3 = nn.Conv2d(16, 19, 5)
        self.fc1 = nn.Linear(19 * 5 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adapt(F.relu(self.conv3(x)))
        x = x.view(-1, 19 * 5 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    '''


## Q2: Vary the Number of Filters ##
### Usually, in a convolutional layer(Pytorch), we can set the number of filters as the number of out_channels. ###
1. Original training accuracy and time costing are 53% and 89.5s respectively.
2. Set the parameter of output_channel from 6 to 3, and the Accuracy of the network on the 10000 test images: 50%, training time is 85.5s.
3. Set the parameter of output_channel from 6 to 9, and the Accuracy of the network on the 10000 test images: 54%, training time is 102.4s.


## Q3: Vary the Learning Rate ##
1. Original training accuracy and time costing are 53% and 89.5s respectively.
2. Set the lr from 0.001 to 0.005, the training accuracy and time costing are 43% and 86.6s respectively.
2. Set the lr from 0.001 to 0.005, the training accuracy and time costing are 43% and 86.6s respectively.


## Q4: Try Different Optimizers ##
1. Original optimizer is SGD: 53%.
2. Set the optimizer to Adam: 55%.
3. Set the optimizer to RMSprop: 54%.
4. Set the optimizer to Adagrad: 36%.


## Q5: Try Batch Normalization ##
1. Add batch normalization in conv2(not include conv1).
2. Add batch normalization in fc1 and fc2(not sure but it should include both fcs)
3. Update code in forward part.
4. Get the result: Accuracy is 56% without step2.
5. Get the result: Accuracy is 45% without step2-fc2.
6. Get the result: Accuracy is 43% with all steps.

