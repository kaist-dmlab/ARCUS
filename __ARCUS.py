import tensorflow as tf
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn import metrics
from CKA import linear_CKA, kernel_CKA
from model.model_utils import ModelGenerator

class ARCUS:
    def __init__(self, args, ModelGenerator):
        self.args = args
        self.args.itr_num = int(args.batch/args.min_batch)
        # set model generator for initializing models
        self.model_generator = ModelGenerator(args)
        

    def _standardize_scores(self, score: float):
        # get standardized anomaly scores
        mean_score = np.mean(score)
        std_score = np.std(score)
        standardized_score = np.array([(k-mean_score)/std_score for k in score])
        return standardized_score
    

    def _merge_models(self, 
                      model1: tf.keras.Model, 
                      model2: tf.keras.Model):
        # merge a previous model and a current model 
        num_batch_sum = model1.num_batch + model2.num_batch
        w1 = model1.num_batch/num_batch_sum
        w2 = model2.num_batch/num_batch_sum
        
        # merge encoder
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

        # merge decoder
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
        
        if self.args.model_type == 'RSRAE':
            model2.A = (model1.A*w1 + model2.A*w2)

        model2.num_batch = num_batch_sum
        return model2

    def _reduce_models_last(self, 
                            x_inp,
                            thred: float, 
                            epoch_num: int):
        # delete similar models for reducing redundancy in a model pool
        latents = []
        for m in self.model_pool:
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
            self.model_pool[max_Idx2] = self._merge_models(self.model_pool[max_Idx1], self.model_pool[max_Idx2])
            self._train_model(self.model_pool[max_Idx2], x_inp, epoch_num) #Train just one epoch to get the latest score info

            self.model_pool.remove(self.model_pool[max_Idx1])       
            if len(self.model_pool) > 1:
                self._reduce_models_last(self.model_pool, x_inp, thred, epoch_num)


    def _train_model(self, 
                     model: tf.keras.Model, 
                     x_inp,
                     epoch_num):
        # train a model in the model pool of ARCUS
        tmp_losses = []
        for _ in range(epoch_num):
            for _ in range(self.args.itr_num):
                min_batch_x_inp = tf.random.shuffle(x_inp)[:self.args.min_batch]
                loss = model.train_step(min_batch_x_inp)
                tmp_losses.append(loss.numpy())
        temp_scores = model.inference_step(x_inp)
        model.last_mean_score = np.mean(temp_scores)
        model.last_max_score = np.max(temp_scores)
        model.last_min_score = np.min(temp_scores)
        model.num_batch = model.num_batch+1
        
        return tmp_losses

    def simulator(self, 
                  dataset: tf.data.Dataset, 
                  inf_type: str, 
                  batch: int, 
                  reliability_thred: float, 
                  similarity_thred: float):
        # Simulator for online anomaly detection
        
        initial_model = self.model_generator.init_model()
        curr_model = deepcopy(initial_model)
        
        self.model_pool = []
        self.model_pool.append(initial_model)
        
        auc_hist   = []
        drift_hist = []
        losses     = []
        all_scores = []
        
        # Scenario for online anomaly detection
        try:    
            for step, (x_inp, y_inp) in enumerate(dataset.batch(batch)):
                # Initial model training
                if step == 0:
                    tmp_losses = self._train_model(initial_model, x_inp, self.args.init_epoch)
                    losses = losses + tmp_losses

                # Inference
                if inf_type == "INC":
                    final_scores = initial_model.inference_step(x_inp)
                else:
                    scores = []
                    model_reliabilities = []
                    for m in self.model_pool:
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
                    curr_model = self.model_pool[curr_model_index]
                    
                    weighted_scores = []
                    for idx in range(len(self.model_pool)):
                        weight = model_reliabilities[idx]
                        weighted_scores.append(self._standardize_scores(scores[idx]) * weight)
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
                    new_model = self.model_generator.init_model()
                    tmp_losses = self._train_model(new_model, x_inp, self.args.init_epoch)
                    losses = losses + tmp_losses
                    self.model_pool.append(new_model)
                    #Merge models
                    self.model_pool = self._reduce_models_last(x_inp, similarity_thred, 1)
                else:
                    tmp_losses = self._train_model(curr_model, x_inp, self.args.intm_epoch)
                    losses = losses + tmp_losses            
        except Exception as e:
            print("At seed: ", self.arg.seed, "Error: ", e)
            return False, None, None

        return True, auc_hist, all_scores
