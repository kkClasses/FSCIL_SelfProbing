from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import logging
import torch.nn as nn

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cpu() for _ in batch] #[_.cuda() for _ in batch]
        logits, logits1 = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        logits1 = logits1[:, :args.base_class]
        loss1 = F.cross_entropy(logits1, train_label)
        acc = count_acc(logits, train_label)
        total_loss = (loss + loss1)/2
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


def base_train1(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()  # This object is used to average the loss
    ta = Averager()  # This object is used to average the accuracy
    model = model.train()  # Set the model in training mode
    tqdm_gen = tqdm(trainloader)  # Initialize the tqdm generator for progress bar
    mw = model.fc.weight[:args.base_class, :]
    mw1 = model.fc1.weight[:args.base_class, :]
    # pairs = torch.combinations(torch.arange(args.base_class), r=2)
    # mw2 = ((mw[pairs[:, 0]] + mw[pairs[:, 1]])/2).detach()
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cpu() for _ in batch]  # Move to CPU (or use .cuda() for GPU)
        embed = model.encode(data)
        #embed = torch.mean(torch.reshape(embed, (5,5,-1)), dim=1)
        # Compute logits
        logits = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
        logits2 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw1, p=2, dim=-1)) * args.temperature
        # logits3 = F.linear(F.normalize(shuffled_embed, p=2, dim=-1), F.normalize(mw2, p=2, dim=-1)) * args.temperature
        # # Compute softmax and entropy for logits2 and logits3
        # epsilon = 1e-10
        # p2 = torch.clamp(torch.nn.functional.softmax(logits2, dim=1), min=epsilon)
        # p3 = torch.clamp(torch.nn.functional.softmax(logits3, dim=1), min=epsilon)
        # # Compute entropy for each
        # entropy2 = torch.reshape(torch.sigmoid(-torch.sum(p2 * torch.log(p2), dim=1, keepdim=True)), (-1, 1))
        # entropy3 = torch.reshape(torch.sigmoid(-torch.sum(p3 * torch.log(p3), dim=1, keepdim=True)), (-1, 1))

        loss1 = F.cross_entropy(logits2, train_label)
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        # Total loss combines classification loss and entropy terms
        total_loss = (loss +loss1)/2 #+ (1 - torch.mean(entropy2)) + torch.mean(entropy3)
        # Log the learning rate and other statistics
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(   'Session 0, epo {}, lrc={:.4f}, total loss={:.4f}, acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        # Update the average loss and accuracy
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        total_loss.backward()  # Only one backward pass here
        optimizer.step()
    model = updatedWeight(initial_weights, model,0)
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cpu() for _ in batch] #[_.cuda() for _ in batch]
            model.mode = 'encoder'  # model.module.mode = 'encoder'
            embedding = model(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    proto_list = []
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        # random_indices = torch.randperm(len(data_index))[:25]
        # embedding_this = embedding_this[random_indices]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    proto_list = torch.stack(proto_list, dim=0)
    model.fc.weight.data[:args.base_class] = proto_list #module.
    return model

def test(model, testloader, epoch, args, session, result_list=None):
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
import pandas as pd
def test2(model, fc2, testloader, epoch, args, session, result_list=None):
    test_class = args.base_class + session * args.way
    model = model.eval()  # Set model to evaluation mode
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    mw = model.fc.weight[:test_class, :]; mw1 = model.fc1.weight[:test_class, :]
    mw2 = fc2.weight[:test_class, :]; mw11 = mw.clone();
    mw11[args.base_class:, :] = torch.normal(0, 0.0001, (test_class-args.base_class, mw1.shape[-1]))
    with (torch.no_grad()):
        for i, (data, test_label) in enumerate(testloader, 1):
            data, test_label = data.cpu(), test_label.cpu()  # Move data to CPU for compatibility
            embed = model.encode(data)
            logits1 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
            logits2 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw1, p=2, dim=-1)) * args.temperature
            logits3 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw2, p=2, dim=-1)) * args.temperature
            logits11 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw11, p=2, dim=-1)) * args.temperature
            #logits = torch.where((test_label.unsqueeze(1) < args.base_class), logits2, logits3)
            # epsilon =1e-10
            # p11 = torch.clamp(torch.nn.functional.softmax(logits11, dim=1), min= epsilon)
            # p2 = torch.clamp(torch.nn.functional.softmax(logits2, dim=1), min= epsilon)
            # p3 = torch.clamp(torch.nn.functional.softmax(logits3, dim=1), min= epsilon)
            mx1,ind1 = torch.max(logits1,dim=1);  #mx11, _ = torch.max(p11, dim=1); mx2,_ = torch.max(p2,dim=1)
            logits = logits1 #torch.where(ind1.unsqueeze(1) < args.base_class, logits2, logits3)
            #logits = torch.where(ind1.unsqueeze(1) < args.base_class,
             #       torch.where(mx1.unsqueeze(1) > mx2.unsqueeze(1), p11, p2),p3)

            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)
            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        #hook_fn.remove()
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if session > 0:
            cm = confmatrix(lgt, lbs)  # save_model_dir
            # if session ==8:
            #     #result_array = cm.detach().numpy()
            #     file_mode = "w" if i == 0 else "a"
            #     with open("confusion_matrix_cifar.csv", mode=file_mode, newline="") as file:
            #         writer = csv.writer(file)
            #         writer.writerows(cm)
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
            logging.info('epo {}, loss={:.4f} acc={:.4f}, acc@5={:.4f} '.format(epoch, vl, va, va5))
            return vl, (seenac, unseenac, va, FN,FP)
        else:
            return vl, va

import torchvision.transforms as T
def dataAug(images, num_aug=5):
    augmentation_pipeline = T.Compose([
        T.RandomResizedCrop(size=(36,36), scale=(0.5, 1)),  # Slight zoom + random crop, cifar(36,36)
        T.RandomHorizontalFlip(p=0.5), ])# 50% chance of flipping
    augmented_images = []
    for img in images:
        #for _ in range(num_aug):  # Apply augmentation `num_aug` times per image
        augmented_images.append(augmentation_pipeline(img))
    return torch.stack(augmented_images)
def updatedWeight(initial_weights, model,session):
    updated_weights = {k: v.clone() for k, v in model.state_dict().items()}  # Get a copy of current weights
    eta = 0.1;     alpha = 100
    new_state_dict = {}
    for (name, initial), (_, updated), layer_name in zip(initial_weights.items(), updated_weights.items(),initial_weights):
        weight_diff = updated-initial
        weight_decay_factor = torch.exp(-torch.abs(alpha * initial))
        modified_weight = initial + eta * weight_decay_factor * weight_diff  #torch.abs()#
        new_state_dict[name] = modified_weight
        if session>0:
            if layer_name =='fc.weight':
                 new_state_dict[name] = updated
    model.load_state_dict(new_state_dict)
    return model

def soft_calibration2(base_protos,cur_protos, args):
    base_protos1 = F.normalize(base_protos, p=2, dim=-1)
    cur_protos1 = F.normalize(cur_protos, p=2, dim=-1)
    updated_protos = cur_protos1
    itr = torch.randint(0,5,(1,))
    for i in range(itr):
        softmax_t = torch.randint(1, 4, (1,)) * 4
        weights = torch.mm(updated_protos, base_protos1.T) * softmax_t #args.softmax_t
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        updated_protos = (1 - args.shift_weight) * updated_protos + args.shift_weight * delta_protos
    return updated_protos



