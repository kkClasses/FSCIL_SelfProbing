import logging
from copy import deepcopy
import torch.nn as nn
from base import Trainer
from dataloader.data_utils import get_dataloader
from utils import *
from .helper import *
from .Network import MYNET
import copy
class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_up_model()

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        #self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        #self.model = self.model.cuda()
        if self.args.model_dir is not None:
            logging.info('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir,  map_location={'cuda:3':'cuda:0'})['params']
        else:
            logging.info('random init params')
            if self.args.start_session > 0:
                logging.info('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

        self.fc2 = nn.Linear(self.model.num_features, self.args.num_classes, bias=False)
        self.radius = torch.ones((self.args.base_class))
        # self.highway1 = Highway1(self.model.num_features)
        # self.jepaDecoder1 = jepaDecoder(self.model.num_features)

        # if self.args.model_dir_self_prob is not None:
        #     logging.info('Loading init parameters from: %s' % self.args.model_dir_self_prob)
        #     self.best_model_dict_selfProb = torch.load(self.args.model_dir_self_prob, map_location={'cuda:3': 'cuda:0'})['params']
        #     self.highway1.load_state_dict(self.best_model_dict_selfProb)

    def train(self,):
        args = self.args
        t_start_time = time.time()
        result_list = [args]
        self.model.load_state_dict(self.best_model_dict)
        self.model.fc1.weight.data.clone()[args.base_class:, :] = 0.0001
        #mw= self.model.fc.weight.data.clone()[:args.base_class, :]
        self.fc2.weight.data = self.model.fc1.weight.data.clone()
        FNN = [];   FPP = []; #low_indices = lowIndices(self.model)
        # self.frozen_model.load_state_dict(self.best_model_dict)
        # for param in self.frozen_model.parameters():
        #     param.requires_grad = False
        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = get_dataloader(args, session)
            self.model.load_state_dict(self.best_model_dict)
            if args.trainTest == 'train':
                if session == 0:  # load base class train img label
                    if not args.only_do_incre:
                        logging.info(f'new classes for this session:{np.unique(train_set.targets)}')
                        self.model = self.model.amplify_protoypes(self.model, args)
                        optimizer, scheduler = get_optimizer(args, self.model)
                        for epoch in range(args.epochs_base):
                            start_time = time.time()
                            #tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                            tl, ta = base_amplify_protoype(self.model, trainloader, optimizer, scheduler, epoch, args)
                            #tl, ta = base_train1(self.model, trainloader, optimizer, scheduler, epoch, args)
                            #tl, ta = base_highway(self.model, self.highway1,  trainloader, optimizer, scheduler, epoch, session, args)
                            #tl, ta = base_jepa(self.model, self.jepaDecoder1, trainloader, optimizer, scheduler, epoch, session, args)
                            tsl, tsa = test(self.model, testloader, epoch, args, session, result_list=result_list)
                            #tsl, tsa = testHighway(self.model, self.highway1, testloader, args, session)
                            # save better model
                            if (tsa * 100) > self.trlog['max_acc'][session]:
                                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                                self.trlog['max_acc_epoch'] = epoch
                                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                                torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                                self.best_model_dict = deepcopy(self.model.state_dict())
                                logging.info('********A better model is found!!**********')
                                logging.info('Saving model to :%s' % save_model_dir)
                                logging.info('best epoch {}, best test acc={:.3f}'.format(
                                self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

                            self.trlog['train_loss'].append(tl)
                            self.trlog['train_acc'].append(ta)
                            self.trlog['test_loss'].append(tsl)
                            self.trlog['test_acc'].append(tsa)
                            lrc = scheduler.get_last_lr()[0]
                            logging.info(
                                'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                    epoch, lrc, tl, ta, tsl, tsa))
                            print('This epoch takes %d seconds' % (time.time() - start_time),
                                '\n still need around %.2f mins to finish this session' % (
                                        (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                            scheduler.step()
                        # Finish base train
                        logging.info('>>> Finish Base Train <<<')
                        result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                            session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))
                    else:
                        logging.info('>>> Load Model &&& Finish base train...')
                        assert args.model_dir is not None

                    if not args.not_data_init:
                        self.model.load_state_dict(self.best_model_dict)
                        self.model.fc1.weight.data = self.model.fc.weight.data.clone()
                        #self.model = self.model.replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                        best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        logging.info('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        torch.save(dict(params=self.model.state_dict()), best_model_dir)

                        self.model.mode = 'avg_cos'
                        tsl, tsa = test(self.model, testloader, 0, args, session, result_list=result_list)
                        if (tsa * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                            logging.info('The new best test acc of base session={:.3f}'.format(
                                self.trlog['max_acc'][session]))

                # incremental learning sessions
                else:
                    logging.info("training session: [%d]" % session)
                    self.model.mode = self.args.new_mode #self.model.module.mode
                    self.model.eval()
                    trainloader.dataset.transform = testloader.dataset.transform
                    if args.soft_mode == 'soft_proto':
                        self.model.update_fc(self.model,trainloader, np.unique(train_set.targets), session)
                        #self.model.soft_calibration(args, session)
                       # optimizer, scheduler = get_optimizer(args, self.model)
                    else:
                        raise NotImplementedError
                    print("unwanted Test:")
                    #tsl, (seenac, unseenac, avgac) = test(self.model, testloader, 0, args, session, result_list=result_list)
                    # update results and save model
                    #tsl, (seenac, unseenac, avgac, FN, FP) = testHighway(self.model, self.highway1,  testloader, args, session)
                    tsl, (seenac, unseenac, avgac)= testAmplify_protoypes(self.model, testloader,  args, session)

                    self.trlog['seen_acc'].append(float('%.3f' % (seenac * 100)))
                    self.trlog['unseen_acc'].append(float('%.3f' % (unseenac * 100)))
                    self.trlog['max_acc'][session] = float('%.3f' % (avgac * 100))
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    logging.info(f"Session {session} ==> Seen Acc:{self.trlog['seen_acc'][-1]} " f"Unseen Acc:{self.trlog['unseen_acc'][-1]} Avg Acc:{self.trlog['max_acc'][session]}")
                    result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
            else:
                logging.info("testing session: [%d]" % session)
                trainloader.dataset.transform = testloader.dataset.transform
                if session==0:
                    #print("replace base fc")
                    #self.model = self.model.replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    #self.best_model_dict = deepcopy(self.model.state_dict())

                    optimizer, scheduler = get_optimizer(args, self.model)

                    #for epoch in range(2):
                          #     tl, ta = session_highway(self.model, self.highway1,  trainloader, optimizer, scheduler, epoch,session,args)
                    # # # # # # # # #   #tl, ta = base_train1(self.model, trainloader, optimizer, scheduler, args)
                    # save_model_dir = os.path.join(args.save_path, 'selfProbe' + str(self.args.dataset) + '.pth')
                    # torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    # torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                    # print(save_model_dir)

                    self.best_model_dict = deepcopy(self.model.state_dict())
                    #self.radius = torch.ones(args.base_class) * 0.25
                    self.model.mode = self.args.new_mode
                    self.model.eval()
                    tsl, tsa = test(self.model, testloader, 0, args, session, result_list=result_list)
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    #self.lowIndices = get_low_weight_indices(self.model, percentage=0.1, num_layers=10)
                else:
                    if args.soft_mode == 'soft_proto':
                        print("update fc")
                        self.model.update_fc(self.model, trainloader, np.unique(train_set.targets), session)
                        #self.model.fc1.weight.data = self.model.fc.weight.data.detach()

                        t1 = args.base_class + (session - 1) * args.way
                        t2 = args.base_class + session * args.way
                        self.fc2.weight.data[t1: t2] = self.model.fc.weight.data[t1: t2]

                        self.best_model_dict = deepcopy(self.model.state_dict())
                    else:
                        raise NotImplementedError
                    if self.args.incremental_training==1:
                        #optimizer, scheduler = get_optimizer(args, self.model)
                        # wt = self.model.fc.weight.data.detach().cpu()
                        # self.model.fc.weight.data = self.fc2.weight.data()
                        #self.binaryClassifedData = binaryClassifiedDataset(trainloader, self.binaryClassifedData, session,args)
                        #if session <3:
                        #self.bc = binaryClassifier_train(self.model, self.bc, self.lowIndices , self.binaryClassifedData, session, args)
                        # for param in list(self.model.parameters()):#[:-5]:
                        #     param.requires_grad = True
                        #self.model = session_train1(self.model, self.fc2, trainloader, optimizer, args.epochs_new, scheduler, session, args)
                        # initial_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
                        #for epoch in range(3):
                            #      #     tl, ta = randomWeights_train(self.model, low_indices, trainloader, optimizer, scheduler, session, epoch, args, initial_weights=initial_weights)
                        #      tl, ta = session_highway(self.model, self.highway1,  trainloader, optimizer, scheduler, epoch,session,args)

                        # self.model.fc.weight.data[:args.base_class] = self.fc2.weight.data.detach().cpu()[:args.base_class]
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        #print(self.model1.alpha,self.model1.beta,self.model1.gamma)
                    self.model.mode = self.args.new_mode  # self.model.module.mode

                    #tsl, (seenac, unseenac, avgac) = testEpisodic(self.model, self.highway1, testloader, args,  session)
                    #tsl, (seenac, unseenac, avgac,FN,FP) = testHighway(self.model, self.highway1, testloader, args, session)
                    #tsl, (seenac, unseenac, avgac) = test(self.model, testloader, 0, args, session, result_list=result_list)
                    #tsl, (seenac, unseenac, avgac, FN, FP) = testNewr(self.model, self.fc2, testloader, args.epochs_new, args, session,result_list=None)
                    tsl, (seenac, unseenac, avgac)= testAmplify_protoypes(self.model, testloader,  args, session)

                    # update results and sa model
                    self.trlog['seen_acc'].append(float('%.3f' % (seenac * 100)))
                    self.trlog['unseen_acc'].append(float('%.3f' % (unseenac * 100)))
                    self.trlog['max_acc'][session] = float('%.3f' % (avgac * 100))
                    #self.trlog['FP'].append(float('%.3f' % (FP * 100)))
                    #self.best_model_dict = deepcopy(self.model.state_dict())
                    logging.info( f"Session {session} ==> Seen Acc:{self.trlog['seen_acc'][-1]} " f"Unseen Acc:{self.trlog['unseen_acc'][-1]} Avg Acc:{self.trlog['max_acc'][session]}")
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
        # Finish all incremental sessions, save results.
        result_list, hmeans = postprocess_results(result_list, self.trlog)

        # save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
        # save_model_dir = os.path.join(args.save_path, 'selfProbe' + str(self.args.dataset) + '.pth')
        # torch.save(dict(params=self.model.state_dict()), save_model_dir)
        # #torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
        # print(save_model_dir)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.beta = nn.Parameter(torch.tensor(0.01))
        self.gamma = nn.Parameter(torch.tensor(0.01))
    def forward(self, x):
        logits1,logits2,logits3 = x
        epsilon = 1e-10
        p1 = torch.clamp(torch.nn.functional.softmax(logits1, dim=1), min=epsilon)
        p2 = torch.clamp(torch.nn.functional.softmax(logits2, dim=1), min=epsilon)
        p3 = torch.clamp(torch.nn.functional.softmax(logits3, dim=1), min=epsilon)
        # Compute entropy for each
        entropy1 = -torch.sum(p1 * torch.log(p1), dim=1, keepdim=True)  # Shape: [batch_size, 1]
        entropy2 = -torch.sum(p2 * torch.log(p2), dim=1, keepdim=True)  # Shape: [batch_size, 1]
        entropy3 = -torch.sum(p3 * torch.log(p3), dim=1, keepdim=True)  # Shape: [batch_size, 1]
        logits = logits1*(torch.sigmoid(entropy1))+0.05*logits2 * (torch.sigmoid(entropy2)) + 0.1*logits3 * (torch.sigmoid(entropy3))
        return logits, [self.alpha,self.beta,self.gamma],0

class BinaryClass(nn.Module):
    def __init__(self, input_dim=64):
        super(BinaryClass, self).__init__()
        self.binaryClassifier = nn.Sequential(
            nn.Linear(input_dim, 128),  # Input size = input_dim, Hidden layer size = 128
            nn.BatchNorm1d(128),  # Batch normalization for 128 features
            #nn.ReLU(),  # Activation function for the hidden layer
            nn.Linear(128, 32),  # Hidden layer size = 32
            nn.BatchNorm1d(32),  # Batch normalization for 32 features
            #nn.ReLU(),  # Activation function for the hidden layer
            nn.Linear(32, 1),  # Output size = 1 (binary classification)
            nn.Sigmoid()  # Sigmoid activation for probability output
        )
    def forward(self, x):
        pred = self.binaryClassifier(x)
        return pred

class meanEmbed(nn.Module):
    def __init__(self, input_dim=64):
        super(meanEmbed, self).__init__()
        self.meanEmbedd = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),  # Replace BatchNorm1d with LayerNorm
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.LayerNorm(input_dim),  # Replace BatchNorm1d with LayerNorm
            nn.ReLU(),
        )
    def forward(self, x):
        return self.meanEmbedd(x)

class Highway1(nn.Module):
    def __init__(self, size):
        super(Highway1, self).__init__()
        self.activation = nn.ReLU()
        self.fc_h = nn.Linear(size, size)  # Nonlinear transformation (H)
        self.fc_t = nn.Linear(size, size)  # Transform gate (T)
        self.sigmoid = nn.Sigmoid()  # Gate activation

    def forward(self, x):
        h,z = x
        z1 = h*z #-z.mean(dim=1, keepdim=True))
        h1 = self.fc_t(z1) #
        T = self.sigmoid( h1)  # Transform gate (values between 0 and 1)
        C = 1 - T  # Carry gate
        h2 = h * T + z1*C # Highway connection
        return self.activation(self.fc_h(h2)) #self.finalEmbed(h,z1) #

class Highway3(nn.Module):
    def __init__(self, size):
        super(Highway1, self).__init__()
        self.activation = nn.ReLU()
        self.fc = nn.Linear(size, size, bias=False)  # nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
    def forward(self, x):
        T1 = self.activation(self.fc(x))  # Gate control
        # T = H * T + x * (1 - T)  # Highway connection
        return T1
class jepaDecoder(nn.Module):
    def __init__(self, size):
        super(jepaDecoder, self).__init__()
        self.activation = nn.ReLU()
        self.fc = nn.Linear(size, size, bias=False)  # nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
    def forward(self, x):
        x11 = self.activation(self.fc(x))
        return x11

class SparseAutoencoderKL(nn.Module):
    def __init__(self, input_size,  sparsity_target=0.1, sparsity_weight=1e-1):
        super(SparseAutoencoderKL, self).__init__()
        self.encoder = nn.Linear(input_size, input_size, bias=False)
        self.decoder = nn.Linear(input_size, input_size, bias=False)
        self.rho_hat_target = sparsity_target
        self.beta = sparsity_weight

    def forward(self, x):
        z = torch.sigmoid(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def kl_divergence_sparsity_loss(self, z,h):
        # Average activation acSross batch, height, and width
        rho_hat = torch.mean(z, dim=1)  # shape: (channels,)
        rho = torch.mean(h, dim=1) #self.rho_hat_target
        rho_hat = torch.clamp(rho_hat, 1e-6, 1 - 1e-6)
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return self.beta * torch.sum(kl)

class Highway2(nn.Module):
    def __init__(self, dim, train_way):
        super(Highway1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=dim[0], out_channels=3 * train_way, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(3 * train_way),
            nn.Conv2d(in_channels=3 * train_way, out_channels=train_way, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=train_way, out_channels=1, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(1),
            nn.Flatten()
        )
    def forward(self, x):
        return self.model(x)