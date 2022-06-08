import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import metrics
from CKA import linear_CKA, kernel_CKA
from model.model_RSRAE import RSRAE
from model.model_RAPP import RAPP
from model.model_DAGMM import DAGMM
tf.keras.backend.set_floatx('float64')

class ARCUS:
    def init_model(self, model_type, layer_size, min_batch, learning_rate, input_dim, random_seed, intrinsic_size, RSRAE_hidden_layer_size):
        if model_type == "RSRAE":
            model =  RSRAE(
                hidden_layer_sizes = [input_dim] + RSRAE_hidden_layer_size, #[input_dim, 32, 64, 128]: default
                learning_rate = learning_rate, 
                bn = True,
                activation = 'relu',
                intrinsic_size = intrinsic_size, 
                random_seed = random_seed, 
                name = "RSRAE")
        elif model_type == "RAPP":
            model = RAPP(hidden_layer_sizes = layer_size,
                learning_rate = learning_rate,
                bn = True,
                activation = 'relu',
                random_seed = random_seed,
                name = 'RAPP')
        elif model_type == "DAGMM":
            model = DAGMM(
                comp_hidden_layer_sizes = layer_size, 
                comp_activation = 'tanh',
                est_hidden_layer_sizes = [3, 10, 4],
                est_activation = 'tanh',
                learning_rate = learning_rate,
                bn = True,
                est_dropout_ratio = 0.5,
                random_seed = random_seed)
        model.num_batch = 0
        return model

    def standardize_scores(self, score):
        mean_score = np.mean(score)
        std_score = np.std(score)
        standardized_score = np.array([(k-mean_score)/std_score for k in score])
        return standardized_score

    def merge_models(self, model1, model2, model_type):
        num_batch_sum = model1.num_batch + model2.num_batch
        w1 = model1.num_batch/num_batch_sum
        w2 = model2.num_batch/num_batch_sum
        
        for layer_idx in range(len(model2.encoder)):
            l_base = model1.encoder[layer_idx]
            l_target = model2.encoder[layer_idx]
            if l_base.name[:5] == 'layer':
                new_weight = (l_base.weights[0]*w1 + l_target.weights[0]*w2)
                new_bias = (l_base.weights[1]*w1 + l_target.weights[1]*w2)
                l_target.set_weights([new_weight, new_bias])
            elif l_base.name[:2] == 'bn':
                new_gamma = (l_base.weights[0]*w1 + l_target.weights[0]*w2)
                new_beta = (l_base.weights[1]*w1 + l_target.weights[1]*w2)
                new_mm = (l_base.weights[2]*w1 + l_target.weights[2]*w2)
                new_mv = (l_base.weights[3]*w1 + l_target.weights[3]*w2)
                l_target.set_weights([new_gamma, new_beta, new_mm, new_mv])

        if model_type == 'RSRAE':
            model2.A = (model1.A*w1 + model2.A*w2)

        for layer_idx in range(len(model2.decoder)):
            l_base = model1.decoder[layer_idx]
            l_target = model2.decoder[layer_idx]
            if l_base.name[:5] == 'layer':
                new_weight = (l_base.weights[0]*w1 + l_target.weights[0]*w2)
                new_bias = (l_base.weights[1]*w1 + l_target.weights[1]*w2)
                l_target.set_weights([new_weight, new_bias])
            elif l_base.name[:2] == 'bn':
                new_gamma = (l_base.weights[0]*w1 + l_target.weights[0]*w2)
                new_beta = (l_base.weights[1]*w1 + l_target.weights[1]*w2)
                new_mm = (l_base.weights[2]*w1 + l_target.weights[2]*w2)
                new_mv = (l_base.weights[3]*w1 + l_target.weights[3]*w2)
                l_target.set_weights([new_gamma, new_beta, new_mm, new_mv])
        
        model2.num_batch = num_batch_sum
        return model2

    def reduce_models_last(self, models, x_inp, model_type, thred, min_batch, epoch_num, itr_num):
        latents = []
        for m in models:
            z = m.get_latent(x_inp)
            latents.append(z.numpy())

        max_CKA = 0
        max_Idx1 = None
        max_Idx2 = len(latents)-1
        for idx1 in range(len(latents)-1):
            CKA = linear_CKA(latents[idx1], latents[max_Idx2])
            if CKA > max_CKA:
                max_CKA = CKA
                max_Idx1 = idx1

        if max_Idx1 != None and max_CKA >= thred:
            models[max_Idx2] = self.merge_models(models[max_Idx1], models[max_Idx2], model_type)
            self.train_model(models[max_Idx2], x_inp, min_batch, epoch_num, itr_num) #Train just one epoch to get the latest score info

            models.remove(models[max_Idx1])       
            if len(models) > 1:
                self.reduce_models_last(models, x_inp, model_type, thred, min_batch, epoch_num, itr_num)

        return models

    def train_model(self, model, x_inp, min_batch, epoch_num, itr_num):
        tmp_losses = []
        for e in range(epoch_num):
            for m in range(itr_num):
                min_batch_x_inp = tf.random.shuffle(x_inp)[:min_batch]
                loss = model.train_step(min_batch_x_inp)
                tmp_losses.append(loss.numpy())
        temp_scores = model.inference_step(x_inp)
        model.last_mean_score = np.mean(temp_scores)
        model.last_max_score = np.max(temp_scores)
        model.last_min_score = np.min(temp_scores)
        model.num_batch = model.num_batch+1
        
        return tmp_losses

    def simulator(self, dataset, model_type, inf_type, batch, min_batch, learning_rate, layer_num, hidden_dim, init_epoch, intm_epoch, reliability_thred, similarity_thred, rand_seed, RSRAE_hidden_layer_size):
        input_dim = dataset.element_spec[0].shape[0]
        itr_num = int(batch/min_batch)
        layer_size = []   
        gap = (input_dim - hidden_dim)/(layer_num-1)
        for idx in range(layer_num):
            layer_size.append(int(input_dim-(gap*idx)))
        #print(layer_size)

        models = []
        initial_model = self.init_model(model_type, layer_size, min_batch, learning_rate, input_dim, rand_seed, hidden_dim, RSRAE_hidden_layer_size)
        models.append(initial_model)
        curr_model = initial_model
        auc_hist = []
        drift_hist = []
        losses = []
        all_scores = []
        
        try:    
            for step, (x_inp, y_inp) in enumerate(dataset.batch(batch)):
                # Initial model training
                if step == 0:
                    tmp_losses = self.train_model(initial_model, x_inp, min_batch, init_epoch, itr_num)
                    losses = losses + tmp_losses

                # Inference
                if inf_type == "INC":
                    final_scores = initial_model.inference_step(x_inp)
                else:
                    scores = []
                    model_reliabilities = []
                    for m in models:
                        scores.append(m.inference_step(x_inp))
                        curr_mean_score = np.mean(scores[-1])
                        curr_max_score = np.max(scores[-1])
                        curr_min_score = np.min(scores[-1])
                        min_score = curr_min_score if curr_min_score < m.last_min_score else m.last_min_score
                        max_score = curr_max_score if curr_max_score > m.last_max_score else m.last_max_score
                        gap = np.abs(curr_mean_score - m.last_mean_score)
                        reliability = np.round(np.exp(-2*gap*gap/((2/batch)*(max_score-min_score)*(max_score-min_score))),4)
                        model_reliabilities.append(reliability)

                    curr_model_index = model_reliabilities.index(max(model_reliabilities))
                    curr_model = models[curr_model_index]

                    weighted_scores = []
                    for idx in range(len(models)):
                        weight = model_reliabilities[idx]
                        weighted_scores.append(self.standardize_scores(scores[idx]) * weight)
                    final_scores = tf.reduce_sum(weighted_scores, 0)
                all_scores = all_scores + list(final_scores.numpy())

                if(tf.reduce_sum(y_inp) > 0):
                    auc = metrics.roc_auc_score(y_inp, final_scores)
                    auc_hist.append(auc)

                #Drift detection
                if inf_type == "INC": 
                    drift = False
                else:
                    pool_reliability = 1-np.prod([1-p for p in model_reliabilities])
                    if pool_reliability < reliability_thred:
                        drift = True
                    else:
                        drift = False

                # Model adaptation
                if(drift):
                    drift_hist.append(step)
                    #Create new model
                    new_model = self.init_model(model_type, layer_size, min_batch, learning_rate, input_dim, rand_seed, hidden_dim, RSRAE_hidden_layer_size)
                    tmp_losses = self.train_model(new_model, x_inp, min_batch, init_epoch, itr_num)
                    losses = losses + tmp_losses
                    models.append(new_model)
                    #Merge models
                    models = self.reduce_models_last(models, x_inp, model_type, similarity_thred, min_batch, 1, itr_num)
                else:
                    tmp_losses = self.train_model(curr_model, x_inp, min_batch, intm_epoch, itr_num)
                    losses = losses + tmp_losses            
        except Exception as e:
            print("At seed: ",rand_seed,"Error: ", e)
            return False, None, None

        return True, auc_hist, all_scores
