
class CatConTranformer_Full_DataLoader(CatConTransformer_Full):
    def __init__(self, input_dim, num_classes, class_sizes, cvae_catdims, cvae_hiddendims, cvae_ld, attn_heads,
                 num_ref_points, transformer_dim, temporal_dim, tr_catdims, cvaeAdv_dims, latAdv_dims,
                 learn_temporal_emb, maxT, device, batch_size=8, beta=1.0, gamma_cvae=1.0, gamma_tr=1.0,
                 train_transfer=False, num_cont=None, other_temporal_dims=None, other_minT=None, other_maxT=None,
                 cont_model='sinusoidal', rec_loss='mse', temporal_uncertainty=False):
        super().__init__(input_dim, num_classes, class_sizes, cvae_catdims, cvae_hiddendims, cvae_ld, attn_heads,
                         num_ref_points, transformer_dim, temporal_dim, tr_catdims, cvaeAdv_dims, latAdv_dims,
                         learn_temporal_emb, maxT, device, batch_size=batch_size, beta=beta, gamma_cvae=gamma_cvae,
                         gamma_tr=gamma_tr, train_transfer=train_transfer, num_cont=num_cont,
                         other_temporal_dims=other_temporal_dims, other_minT=other_minT, other_maxT=other_maxT,
                         cont_model=cont_model, rec_loss=rec_loss, temporal_uncertainty=temporal_uncertainty)
        self.dl = None

    def train_model(self, dl, reload, mfile, transfer_dl=None):
        self.dl = dl

        if self.train_transfer:
            assert transfer_dl is not None

        mdir = '/'.join(mfile.split('/')[:-1]) + '/'
        if not os.path.exists(mdir):
            os.makedirs(mdir)

        if reload:
            self.model.load_state_dict(torch.load(mfile, map_location=self.device))
        else:
            n_epochs = 9000
            losses = {}
            val_losses = {}

            tr_opt = torch.optim.Adam(lr=1e-3, params=self.model.parameters())
            adv_cvae_opt = torch.optim.Adam(lr=1e-5, params=self.model.cvae_adv.parameters())
            adv_lat_opt = torch.optim.Adam(lr=1e-5, params=self.model.latent_adv.parameters())
            saved_model = False

            for ep in tqdm(range(n_epochs)):
                if ep > 0 and check_for_early_termination(val_losses['loss'], min_epochs=100):
                    print('Terminating after {} epochs'.format(ep))
                    self.model.load_state_dict(torch.load(mdir + 'tmp_model.pth', map_location=self.device))
                    torch.save(self.model.state_dict(), mfile)
                    saved_model = True
                    break
                ep_losses = {}
                ep_val_losses = {}

                for expr, dr, ce, dsg, time, mask in dl.get_data_loader('train'):
                    N, T, M = expr.shape
                    dr = dr[:, 0]
                    ce = ce[:, 0]

                    # freeze adversaries and train cvae + transformer
                    self.model.cvae_adv.eval()
                    self.model.latent_adv.eval()
                    xhat, mu, logvar, adv_cvae, adv_tr = self.model(
                        expr.float(), 
                        dsg.float(),
                        [torch.stack([dr for _ in range(T)], dim=1).long(),torch.stack([ce for _ in range(T)], dim=1).long()],
                        mask.float(), 
                        other_ts=[time.float()])
                    loss_transformer = self.model.loss_fn(
                        expr.float(), [dr.long(), ce.long()], xhat, mu, logvar, adv_cvae, adv_tr,
                        beta=self.beta, adv_cvae_weight=self.gamma_cvae, adv_tr_weight=self.gamma_tr)
                    for k in loss_transformer:
                        if k not in ep_losses:
                            ep_losses[k] = []
                        ep_losses[k].append(loss_transformer[k].item())
                    tr_opt.zero_grad()
                    loss_transformer['loss'].backward()
                    tr_opt.step()

                    # now switch and train adversaries as needed
                    self.model.cvae_adv.train()
                    self.model.latent_adv.train()

                    if self.gamma_cvae != 0:  # train the cvae adversary
                        _, _, _, adv_cvae, _ = self.model(
                            expr.float(), dsg.float(),
                            [torch.stack([dr for _ in range(T)], dim=1).long(),
                             torch.stack([ce for _ in range(T)], dim=1).long()],
                            mask.float(), other_ts=[time.float()])
                        loss_cvaeAdv = self.model.cvae_adv.loss_fn(adv_cvae, [dr.long(), ce.long()])
                        adv_cvae_opt.zero_grad()
                        loss_cvaeAdv.backward()
                        adv_cvae_opt.step()

                    if self.gamma_tr != 0:  # train the transformer adversary
                        _, _, _, _, adv_tr = self.model(expr.float(), dsg.float(),
                            [torch.stack([dr for _ in range(T)], dim=1).long(),
                             torch.stack([ce for _ in range(T)], dim=1).long()],
                            mask.float(), other_ts=[time.float()])
                        loss_latAdv = self.model.latent_adv.loss_fn(adv_tr, [dr.long(), ce.long()])
                        adv_lat_opt.zero_grad()
                        loss_latAdv.backward()
                        adv_lat_opt.step()

                # potentially repeat for transfer task!
                if self.train_transfer:
                    for sources, targets in transfer_dl:
                        sexpr, sdr, sce, sdsg, stime, smask = sources
                        texpr, tdr, tce, tdsg, ttime, tmask = targets
                        sdr = sdr[:, 0]
                        sce = sce[:, 0]
                        tdr = tdr[:, 0]
                        tce = tce[:, 0]
                        N, T, M = sexpr.shape
                        
                        self.model.cvae_adv.eval()
                        self.model.latent_adv.eval()
                        xgen, mu, logvar, adv_cvae, adv_tr = self.model.generate(
                            sexpr.float(), sdsg.float(), tdsg.float(),
                            [torch.stack([sdr for _ in range(T)], dim=1).long(),
                             torch.stack([sce for _ in range(T)], dim=1).long()],
                            [torch.stack([tdr for _ in range(T)], dim=1).long(),
                             torch.stack([tce for _ in range(T)], dim=1).long()],
                            smask.float(), old_other_ts=[stime.float()], new_other_ts=[ttime.float()],
                            compute_adv=True)
                        loss_transformer = self.model.loss_fn(
                            texpr.float(), [sdr.long(), sce.long()], xgen, mu, logvar, adv_cvae, adv_tr,
                            beta=self.beta, adv_cvae_weight=self.gamma_cvae, adv_tr_weight=self.gamma_tr)
                        for k in loss_transformer:
                            tr_k = 'transfer_{}'.format(k)
                            if tr_k not in ep_losses:
                                ep_losses[tr_k] = []
                            ep_losses[tr_k].append(loss_transformer[k].item())
                        tr_opt.zero_grad()
                        loss_transformer['loss'].backward()
                        tr_opt.step()

                        # unfreeze adversaries to train them as needed
                        self.model.cvae_adv.train()
                        self.model.latent_adv.train()

                        if self.gamma_cvae != 0:
                            _, _, _, adv_cvae, _ = self.model.generate(
                                sexpr.float(), sdsg.float(), tdsg.float(),
                                [torch.stack([sdr for _ in range(T)], dim=1).long(),
                                 torch.stack([sce for _ in range(T)], dim=1).long()],
                                [torch.stack([tdr for _ in range(T)], dim=1).long(),
                                 torch.stack([tce for _ in range(T)], dim=1).long()],
                                smask.float(), old_other_ts=[stime.float()], new_other_ts=[ttime.float()],
                                compute_adv=True)
                            loss_cvaeAdv = self.model.cvae_adv.loss_fn(adv_cvae, [sdr.long(), sce.long()])
                            adv_cvae_opt.zero_grad()
                            loss_cvaeAdv.backward()
                            adv_cvae_opt.step()

                        if self.gamma_tr != 0:
                            _, _, _, _, adv_tr = self.model.generate(
                                sexpr.float(), sdsg.float(), tdsg.float(),
                                [torch.stack([sdr for _ in range(T)], dim=1).long(),
                                 torch.stack([sce for _ in range(T)], dim=1).long()],
                                [torch.stack([tdr for _ in range(T)], dim=1).long(),
                                 torch.stack([tce for _ in range(T)], dim=1).long()],
                                smask.float(), old_other_ts=[stime.float()], new_other_ts=[ttime.float()],
                                compute_adv=True)
                            loss_latAdv = self.model.latent_adv.loss_fn(adv_tr, [sdr.long(), sce.long()])
                            adv_lat_opt.zero_grad()
                            loss_latAdv.backward()
                            adv_lat_opt.step()

                # finally, run validation check
                self.model.eval()
                for expr, dr, ce, dsg, time, mask in dl.get_data_loader('val'):
                    dr = dr[:, 0]
                    ce = ce[:, 0]
                    N, T, M = expr.shape
                    xhat, mu, logvar, adv_cvae, adv_tr = self.model(
                        expr.float(), dsg.float(),
                        [torch.stack([dr for _ in range(T)], dim=1).long(),
                         torch.stack([ce for _ in range(T)], dim=1).long()],
                        mask.float(), other_ts=[time.float()])
                    loss_transformer = self.model.loss_fn(
                        expr.float(), [dr.long(), ce.long()], xhat, mu, logvar, adv_cvae, adv_tr)
                    for k in loss_transformer:
                        if k not in ep_val_losses:
                            ep_val_losses[k] = []
                        ep_val_losses[k].append(loss_transformer[k].item())
                self.model.train()

                for k in ep_losses:
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(np.mean(ep_losses[k]))
                for k in ep_val_losses:
                    if k not in val_losses:
                        val_losses[k] = []
                    val_losses[k].append(np.mean(ep_val_losses[k]))
                if val_losses['loss'][-1] == min(val_losses['loss']):  # best epoch so far!
                    torch.save(self.model.state_dict(), mdir + 'tmp_model.pth')

            if not saved_model:
                self.model.load_state_dict(torch.load(mdir + 'tmp_model.pth', map_location=self.device))
                torch.save(self.model.state_dict(), mfile)

            loss_file = mfile.split('.pth')[0] + '_losses.png'
            plt.figure(figsize=(6, 12) if not self.train_transfer else (6, 24))
            for i, k in enumerate(losses):
                if not self.train_transfer:
                    ax = plt.subplot(3, 2, i+1)
                else:
                    ax = plt.subplot(6, 2, i+1)
                ax.plot(losses[k], label='train {}'.format(k))
                if k in val_losses:  # not a trainsfer task
                    ax.plot(val_losses[k], label='val {}'.format(k))
                plt.legend()
            plt.savefig(loss_file, bbox_inches='tight')
            plt.close()

        self.model.eval()

    def reconstruct(self):
        rec_expr = []
        rec_mask = []
        for expr, dr, ce, dsg, time, mask in self.dl.get_data_loader('train'):
            dr = dr[:, 0]
            ce = ce[:, 0]
            N, T, M = expr.shape
            xhat = self.model(
                expr.float(), dsg.float(),
                [torch.stack([dr for _ in range(T)], dim=1).long(),
                 torch.stack([ce for _ in range(T)], dim=1).long()],
                mask.float(), other_ts=[time.float()])[0]
            rec_expr.append(xhat)
            rec_mask.append(mask)
        rec_expr = torch.cat(rec_expr)
        rec_mask = torch.cat(rec_mask)
        return torch.masked_select(rec_expr, rec_mask.view(-1, T, 1).bool()).view(-1, M)

    def generate(self, gen_dl):
        gen_expr = []
        gen_mask = []
        for sources, targets in gen_dl:
            sexpr, sdr, sce, sdsg, stime, smask = sources
            texpr, tdr, tce, tdsg, ttime, tmask = targets
            sdr = sdr[:, 0]
            sce = sce[:, 0]
            tdr = tdr[:, 0]
            tce = tce[:, 0]
            N, T, M = sexpr.shape
            xhat = self.model.generate(
                sexpr.float(), sdsg.float(), tdsg.float(),
                [torch.stack([sdr for _ in range(T)], dim=1).long(),
                 torch.stack([sce for _ in range(T)], dim=1).long()],
                [torch.stack([tdr for _ in range(T)], dim=1).long(),
                 torch.stack([tce for _ in range(T)], dim=1).long()],
                smask.float(), old_other_ts=[stime.float()], new_other_ts=[ttime.float()])
            
            gen_expr.append(xhat)
            gen_mask.append(tmask)
        gen_expr = torch.cat(gen_expr, dim=0)
        gen_mask = torch.cat(gen_mask, dim=0)
        return torch.masked_select(gen_expr, gen_mask.view(-1, T, 1).bool()).view(-1, M)
# The other important thing to note here is how you set up your dataset. This code would expect, in your example, an input sequence of (image, gene, slice, time) , where the image[i] corresponds to slice[i] and time[i]. As a more concrete example, if I have slice 3 measured at times 2 and 4, and slice 4 measured at times 3 and 4 (for the same gene), then the input sequence could look something like [(im_32, gene, 3, 2), (im_34, gene, 3, 4), (im_43, gene, 4, 3), (im_44, gene, 4, 4)] . Does that make sense?