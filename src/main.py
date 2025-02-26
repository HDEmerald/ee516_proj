import network
import loader

# Load in MNIST Data for training
(train_data, valid_data, test_data) = loader.load_data_wrapper()

# List of sizes of each layer of nodes
# The list of layers for the Neocognitron goes:
# U0, U_S1, U_C1, U_S2, U_C2, U_S3, U_C3, U_S4, U_C4
# The S layers get an additional plane for the V cells
sizes = [19*19, 
         19*19*(12+1), 
         21*21*8, 
         21*21*(80+1), 
         13*13*33, 
         13*13*(97+1), 
         7*7*64, 
         3*3*(47+1), 
         1*35]

# Create neural network
nn = network.Network(sizes)
