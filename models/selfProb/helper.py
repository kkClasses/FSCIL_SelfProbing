import sympy
import torch
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import logging
import torch.nn as nn

def base_preTrain(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cpu() for _ in batch] #[_.cuda() for _ in batch]
        logits, logits1 = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        #logits1 = logits1[:, :args.base_class]
        #loss1 = F.cross_entropy(logits1, train_label)
        acc = count_acc(logits, train_label)
        total_loss = loss #+ loss1
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def base_selfProbe(model, model1, trainloader, optimizer, scheduler,epoch,session, args):
    tl = Averager()  # This object is used to average the loss
    ta = Averager()  # This object is used to average the accuracy
    test_class = args.base_class + session * args.way
    model = model.train()  # Set the model in eval mode
    model1 = model1.train()

    embeddings = innerFeature(model, args)
    mw = model.fc.weight[:test_class, :]
    tqdm_gen = tqdm(trainloader)  # Initialize the tqdm generator for progress bar

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cpu() for _ in batch]  # Move to CPU (or use .cuda() for GPU)
        embeddings.clear()
        H = model.encode(data).detach()
        z = embeddings['kk40']
        T =   model1([H,z]) 

        logits = F.linear(F.normalize(H, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
        rho_hat = torch.mean(T);  sparsity_target = torch.mean(H) #*torch.normal(1.0,0.1,(1,)).item()
        loss2 = kl_divergence(rho_hat, sparsity_target=sparsity_target)

        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        total_loss =  loss  + loss2 #+F.cross_entropy(logits2, train_label) #
       # Log the learning rate and other statistics
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(   'Session {}, epo {}, lrc={:.4f}, total loss={:.4f}, acc={:.4f}'.format(session,epoch, lrc, total_loss.item(), acc))
        # Update the average loss and accuracy
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        total_loss.backward()  # Only one backward pass here
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    #hook_fn.remove()
    return tl, ta #low_indices

def kl_divergence(rho_hat,sparsity_target=0.1):
    """KL divergence between desired sparsity and actual activation"""
    rho = sparsity_target
    rho_hat = torch.clamp(rho_hat, 1e-6, 1 - 1e-6)
    return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))


def testPreTrain(model, testloader, epoch, args, session, result_list=None):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    vc = Averager()
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cpu() for _ in batch]  # [_.cuda() for _ in batch]
            logits, logits1 = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)
            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        #vc = vc.item()
        logging.info('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))
        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if (session==8):
             writeFile(model.fc.weight.detach())

        if session > 0:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs)  # save_model_dir
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            result_list.append(f"Seen Acc:{seenac}  Unseen Acc:{unseenac}")
            return vl, (seenac, unseenac, va)
        else:
            return vl, va

def testselfProbe(model, model1, testloader, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()  # Set model to evaluation mode
    model1 = model1.eval(); #model2 = model2.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    mw = model.fc.weight[:test_class, :];
    mw1 = model.fc1.weight[:test_class, :];
    embeddings = innerFeature(model,  args)

    with (torch.no_grad()):
        for i, (data, test_label) in enumerate(testloader, 1):
            data, test_label = data.cpu(), test_label.cpu()  # Move data to CPU for compatibility
            embeddings.clear()
            z1 = model.encode(data).detach()
            z2 = embeddings['kk40']
            #zz = finalEmbed (H,z);
            z4 = model1( [z1,z2]) #model1(H*z) #
            z5 = z1*args.alpha + z4*(1-args.alpha);
            logits1 = F.linear(F.normalize(z1, p=2, dim=-1), F.normalize(mw1, p=2, dim=-1)) * args.temperature
            logits2 = F.linear(F.normalize(z5, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
            mx, ind = torch.max(logits2, dim=1)
         
            is_base = ind.unsqueeze(1) < args.base_class
            logits = torch.where(is_base, logits1, logits2)

            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)
            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
            if session==8:
                writeFile(z1,z4,i,args)
        #hook_fn.remove()
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if session > 0:
            cm = confmatrix(lgt, lbs)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            p = np.sum(cm[:args.base_class, :test_class])
            fn = np.sum(cm[:args.base_class, args.base_class:test_class])
            n = np.sum(cm[args.base_class:test_class, :test_class])
            fp = np.sum(cm[args.base_class:test_class, :args.base_class])
            FP = np.round(np.mean(fp/n)*100,2) #- np.diag(cm)[args.base_class:])  # Sum of columns excluding diagonal
            FN = np.round(np.mean(fn/p)*100,2)
            #print(cm.shape,FP,FN)
            #result_list.append(f"Seen Acc:{seenac}  Unseen Acc:{unseenac} ")
            logging.info('loss={:.4f} acc={:.4f}, acc@5={:.4f} '.format( vl, va, va5))
            return vl, (seenac, unseenac, va, FN,FP)
        else:
            return vl, va

def innerFeature(model, args):
    embeddings = {};
    def hook_fn(module, input, output):
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(output.size(0), -1)
        embeddings['kk40'] = output
    if args.dataset == 'cifar100':
        relu_layer = model.encoder.layer3[0].relu
        relu_layer.register_forward_hook(hook_fn)
    else:
        relu_layer = model.encoder.layer4[0].relu
        relu_layer.register_forward_hook(hook_fn)
    return embeddings

import csv
def writeFile(z1,z4,i,args):
        result_tensor = torch.cat([z1[:10,:], z4[:10,:]], dim=1)
        result_array = result_tensor.detach().numpy()
        file_mode = "w" if i == 0 else "a"
        #print("file saved")
        with open("results/embed_selfProb"+args.dataset+"_.csv", mode=file_mode, newline="") as file:
            writer = csv.writer(file)
            writer.writerows(result_array)


