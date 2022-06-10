from model.model_RSRAE import RSRAE
from model.model_RAPP import RAPP
from model.model_DAGMM import DAGMM
'''
Create your own ModelGenerator.
1. Initialize arguments of your AE model in __init__ of ModelGenerator.
2. Instantiate your model class with the arguments and return the instance of your model.
'''
class ModelGenerator:
    def __init__(self, args):

        self.layer_size = []   
        gap = (args.input_dim - args.hidden_dim)/(args.layer_num-1)
        for idx in range(args.layer_num):
            self.layer_size.append(int(args.input_dim-(gap*idx)))

        self.model_type = args.model_type
        self.learning_rate = args.learning_rate
        
        # For RSRAE
        if args.model_type == 'RSRAE':
            self.input_dim = args.input_dim
            self.intrinsic_size = args.intrinsic_size
            self.RSRAE_hidden_layer_size = args.RSRAE_hidden_layer_size
        
    def init_model(self):
        # initialize ARCUS framework
        if self.model_type == "RSRAE":
            model =  RSRAE(hidden_layer_sizes = [self.input_dim] + self.RSRAE_hidden_layer_size, #[input_dim, 32, 64, 128]: default
                            activation = 'relu',
                            intrinsic_size = self.intrinsic_size, 
                            learning_rate = self.learning_rate, 
                            bn = True,
                            name = "RSRAE")

        elif self.model_type == "RAPP":
            model = RAPP(hidden_layer_sizes = self.layer_size,
                            activation = 'relu',
                            learning_rate = self.learning_rate,
                            bn = True,
                            name = 'RAPP')

        elif self.model_type == "DAGMM":
            model = DAGMM(comp_hidden_layer_sizes = self.layer_size, 
                            comp_activation = 'tanh',
                            est_hidden_layer_sizes = [3, 10, 4],
                            est_activation = 'tanh',
                            est_dropout_ratio = 0.5,
                            learning_rate = self.learning_rate,
                            bn = True,
                            name='DAGMM')

        model.num_batch = 0
        return model

