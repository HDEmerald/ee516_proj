import network
import loader

# Load in MNIST Data for training
(train_data, valid_data, test_data) = loader.load_data_wrapper()

# List of sizes of each layer of nodes
sizes = [19*19, 
         19*19*12, 
         21*21*8, 
         21*21*80, 
         13*13*33, 
         13*13*97, 
         7*7*64, 
         3*3*47, 
         1*35]

# Create neural network
nn = network.Network(sizes)

# Train the neural network
nn.SGD(train_data, 25, 10, 0.05)

# Evaluate the model and print the correct results
print(str(nn.evaluate(test_data)) + 
      " correct results out of " + 
      str(len(test_data)))