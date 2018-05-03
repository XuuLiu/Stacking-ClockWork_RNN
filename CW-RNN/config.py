
class Config(object):

    output_dir = "./output/"

    # Clockwork RNN parameters
    periods     = [1, 2, 4, 8, 16, 32, 64, 128]#, 256]
    num_steps   = 100 #no use, equals to how many steps backwards
    num_input   = 2 #no use, equals to the dimensions of x_train
    num_hidden  = 160 #num_hidden%periods=0, every period should have the same number of cells
    num_output  = 2 # no use, equals to the dimensions of y_train

    # Optmization parameters
    num_epochs          = 120
    batch_size          = 55 #256
    optimizer           = "adam" #"rmsprop"  adam
    max_norm_gradient   = 10.0

    # Learning rate decay schedule
    learning_rate       = 1e-3
    learning_rate_decay = 0.975
    learning_rate_step  = 1000
    learning_rate_min   = 1e-5



