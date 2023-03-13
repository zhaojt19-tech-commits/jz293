import json
import torch
import numpy as np
from torch import nn
from VAE_Module import *
from MLP_Module import *
from Loss_Module import *

class ComPert(torch.nn.Module):
    """
    Our main module, the ComPert autoencoder
    """

    def __init__(
            self,
            image_H, 
            image_W, 
            input_channels,
            output_channels, 
            device="cuda:0",
            seed=0,
            patience=5,
            doser_type='logsigm',
            hparams="", 
            name_id=1, 
            shape=[]):
        super(ComPert, self).__init__()
        
        # set generic attributes
        self.H = image_H
        self.W = image_W
        
        # 两个参数，存疑
        num_drugs = shape[0]
        num_cell_types =1

        self.device = device
        self.seed = seed

        # early-stopping 用于调整学习率，暂时先不考虑
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters, 写完整体逻辑以后再调整
        self.set_hparams_(seed, hparams)

        # set models

        # 编码器部分，用于处理图像
        self.VAE = VAE(image_H, image_W, input_channels, output_channels, self.hparams['dim'], device, name_id)
        
        self.adversary_drugs = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_drugs])

        self.adversary_cell_types = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_cell_types])

        # set dosers
        self.doser_type = doser_type
        if doser_type == 'mlp':
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP([1] +
                        [self.hparams["dosers_width"]] *
                        self.hparams["dosers_depth"] +
                        [1],
                        batch_norm=False))
        else:
            self.dosers = GeneralizedSigmoid(num_drugs, self.device, nonlin=doser_type)

        self.drug_embeddings = torch.nn.Embedding(
            num_drugs, self.hparams["dim"])
        self.cell_type_embeddings = torch.nn.Embedding(
            num_cell_types, self.hparams["dim"])

        # losses 
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.loss_adversary_cell_types = torch.nn.CrossEntropyLoss()
        self.iteration = 0
        
        # 将模型转移到GPU上
        self.to(self.device)

        # optimizers
        self.optimizer_autoencoder = torch.optim.Adam(
            list(self.VAE.parameters()) +
            list(self.drug_embeddings.parameters()) +
            list(self.cell_type_embeddings.parameters()),
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"])

        self.optimizer_adversaries = torch.optim.Adam(
            list(self.adversary_drugs.parameters()) +
            list(self.adversary_cell_types.parameters()),
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"])

        self.optimizer_dosers = torch.optim.Adam(
            self.dosers.parameters(),
            lr=self.hparams["dosers_lr"],
            weight_decay=self.hparams["dosers_wd"])

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"])

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"])

        self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
            self.optimizer_dosers, step_size=self.hparams["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

        # # 用于数据归一化
        # self.softmax = nn.Softmax(dim=1)

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        default = (seed == 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            "dim": 128 if default else
            int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else
            int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else
            int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-4 if default else
            float(10**np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else
            float(10**np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else
            int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else
            int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else
            int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else
            int(np.random.choice([2, 3, 4])),
            "reg_adversary": 0 if default else
            float(10**np.random.uniform(-2, 2)),
            "penalty_adversary": 1 if default else
            float(10**np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-4 if default else
            float(10**np.random.uniform(-4, -2)),
            "adversary_lr": 1e-4 if default else
            float(10**np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else
            float(10**np.random.uniform(-8, -4)),
            "adversary_wd": 1e-7 if default else
            float(10**np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else
            int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else
            int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 30 if default else
            int(np.random.choice([15, 25, 45])),
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def compute_drug_embeddings_(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """

        if self.doser_type == 'mlp':
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embeddings.weight
        else:
            return self.dosers(drugs) @ self.drug_embeddings.weight

    def predict(self, genes, labels, input_size, num_classes, return_latent_basal=False):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """
        genes, drugs, cell_types = self.move_inputs_(genes, labels, input_size, num_classes)

        mu, log_var = self.VAE.encode(genes.reshape(input_size, 1, self.H, self.W), input_size)
        latent_basal = self.VAE.sample_z(mu, log_var)
        drug_emb = self.compute_drug_embeddings_(drugs)
        # cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        # print(latent_basal.shape)
        # print(drug_emb.shape)
        # print(cell_emb.shape)
        # print(cell_emb)

        latent_treated = latent_basal + drug_emb
        gene_reconstructions = self.VAE.decode(latent_treated, input_size)
        gene_reconstructions = gene_reconstructions.reshape(input_size, 1, self.H, self.W)
        # print([torch.min(gene_reconstructions).item(), torch.max(gene_reconstructions).item()])
        # print([torch.min(genes).item(), torch.max(genes).item()])
        
        # 数据归一化
        # gene_reconstructions = gene_reconstructions.reshape(input_size, -1)
        # max_list = torch.max(gene_reconstructions, 1, keepdim=True).values
        # min_list = torch.min(gene_reconstructions, 1, keepdim=True).values
        # gene_reconstructions = (gene_reconstructions - min_list) / (max_list - min_list)
        # gene_reconstructions = gene_reconstructions.reshape(input_size, 1, self.H, self.W)
        
        # # 数据归一化
        # gene_reconstructions = self.softmax(gene_reconstructions.reshape(input_size, -1)).reshape(input_size, 1, self.H, self.W)
        
        if return_latent_basal:
            return gene_reconstructions, latent_basal

        return gene_reconstructions

    def update(self, genes, labels, input_size, num_classes):
        """
        Update ComPert's parameters given a minibatch of genes, drugs, and
        cell types.
        """
        genes, drugs, cell_types = self.move_inputs_(genes, labels, input_size, num_classes)

        gene_reconstructions, latent_basal = self.predict(
            genes, labels, input_size, num_classes, return_latent_basal=True)
        genes = genes.reshape(input_size, 1, self.H, self.W)
        reconstruction_loss = self.VAE.compute_loss(genes, gene_reconstructions, 0, 0, 0)

        adversary_drugs_predictions = self.adversary_drugs(latent_basal)
        adversary_drugs_loss = self.loss_adversary_drugs(
            adversary_drugs_predictions, drugs.gt(0).float())

        # adversary_cell_types_predictions = self.adversary_cell_types(
        #     latent_basal)
        # adversary_cell_types_loss = self.loss_adversary_cell_types(
        #     adversary_cell_types_predictions, cell_types.argmax(1))

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.Tensor([0])
        adversary_cell_types_penalty = torch.Tensor([0])
        a = False
        # if self.hparams['adversary_steps'] != 0 and self.iteration % self.hparams["adversary_steps"]:
        if(a):
#             print('taking adversary on', self.iteration)
            adversary_drugs_penalty = torch.autograd.grad(
                adversary_drugs_predictions.sum(),
                latent_basal,
                create_graph=True)[0].pow(2).mean()

            # adversary_cell_types_penalty = torch.autograd.grad(
            #     adversary_cell_types_predictions.sum(),
            #     latent_basal,
            #     create_graph=True)[0].pow(2).mean()

            self.optimizer_adversaries.zero_grad()
            loss_term_adversary = (adversary_drugs_loss +
             self.hparams["penalty_adversary"] *
             adversary_drugs_penalty
             )
            loss_term = torch.Tensor([0])
            loss_term_adversary.backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            self.optimizer_dosers.zero_grad()
            loss_term = (reconstruction_loss -
             self.hparams["reg_adversary"] *
             adversary_drugs_loss
             )
            loss_term_adversary = torch.Tensor([0])
            loss_term.backward()
            self.optimizer_autoencoder.step()
            self.optimizer_dosers.step()

        self.iteration += 1

        return {
            'loss': loss_term.item(),
            'loss_adversary': loss_term_adversary.item(), 
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_cell_types": adversary_cell_types_penalty.item()
        }, gene_reconstructions
    
    def move_inputs_(self, genes, labels, input_size, num_classes):
        """
        Move minibatch tensors to CPU/GPU and transform the labels to one-hot.
        """
        label_double = labels.reshape(input_size, -1)
        label0 = label_double[:, 0].long()
        index = np.linspace(0, input_size - 1, input_size).astype("long")
        one_hot_0 = torch.zeros(input_size, num_classes[0]).long()
        one_hot_1 = torch.zeros(input_size, 1)
        one_hot_0[[index, label0]] = labels[:, 1].long()

        # print(one_hot_0.shape)
        # print(one_hot_0[[index, label0]])

        genes = genes.to(self.device)
        drugs = one_hot_0.to(self.device)
        cell_types = one_hot_1.to(self.device)
        return genes, drugs, cell_types


    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        self.scheduler_adversary.step()
        self.scheduler_dosers.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience
    
    def MSE_ori(self, x, recon):
        loss_fn = torch.nn.MSELoss(reduction = 'mean')
        recon_loss = loss_fn(recon, x)

        return recon_loss
    
    def MSE_sample(self, x, recon):
        loss_fn = torch.nn.MSELoss(reduction = 'none')
        recon_loss = loss_fn(recon, x)

        input_size = x.shape[0]
        recon_loss = recon_loss.reshape(input_size, -1)
        recon_loss = recon_loss.mean(axis=1, keepdim=False)

        return recon_loss

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for ComPert
        """

        return self.set_hparams_(self, 0, "")





