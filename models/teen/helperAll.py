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


import torch


def split_labels(train_label):
    unique_labels = train_label.unique()  # Get unique class labels
    num_classes = len(unique_labels)
    # Shuffle labels randomly
    shuffled_labels = unique_labels[torch.randperm(num_classes)]
    # Split into three groups
    split1, split2 = num_classes // 3, (2 * num_classes) // 3
    labels_1, labels_2, labels_3 = shuffled_labels[:split1], shuffled_labels[split1:split2], shuffled_labels[split2:]
    # Get indices for each label set
    mask_1 = torch.isin(train_label, labels_1)
    mask_2 = torch.isin(train_label, labels_2)
    mask_3 = torch.isin(train_label, labels_3)
    indices_1 = torch.nonzero(mask_1, as_tuple=True)[0]
    indices_2 = torch.nonzero(mask_2, as_tuple=True)[0]
    indices_3 = torch.nonzero(mask_3, as_tuple=True)[0]
    return indices_1, indices_2, indices_3

def base_train1(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()  # This object is used to average the loss
    ta = Averager()  # This object is used to average the accuracy
    model = model.train()  # Set the model in training mode
    tqdm_gen = tqdm(trainloader)  # Initialize the tqdm generator for progress bar
    # Prepare weight matrices
    mw = model.fc.weight[:args.base_class, :]
    mw1 = model.fc1.weight[:args.base_class, :]
    pairs = torch.combinations(torch.arange(args.base_class), r=2)
    mw2 = ((mw[pairs[:, 0]] + mw[pairs[:, 1]])/2).detach()
    # Loop through the batches in the training data
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cpu() for _ in batch]  # Move to CPU (or use .cuda() for GPU)
        # Get embeddings from the model
        embed = model.encode(data)
        # Compute logits
        logits = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
        # Split the labels into different indices (idx1, idx2, idx3)
        idx1, idx2, idx3 = split_labels(train_label)
        min_size = min(len(idx1), len(idx2), len(idx3))
        idx1, idx2, idx3 = idx1[:min_size], idx2[:min_size], idx3[:min_size]
        # Compute shuffled embeddings for contrastive loss
        shuffled_embed = model.encode((data[idx1] + data[idx2]) / 2 ) # or other combinations of embeddings
        # Compute logits for shuffled embeddings
        logits2 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw2, p=2, dim=-1)) * args.temperature
        logits3 = F.linear(F.normalize(shuffled_embed, p=2, dim=-1), F.normalize(mw2, p=2, dim=-1)) * args.temperature
        # Compute softmax and entropy for logits2 and logits3
        epsilon = 1e-10
        p2 = torch.clamp(torch.nn.functional.softmax(logits2, dim=1), min=epsilon)
        p3 = torch.clamp(torch.nn.functional.softmax(logits3, dim=1), min=epsilon)
        # Compute entropy for each
        entropy2 = torch.reshape(torch.sigmoid(-torch.sum(p2 * torch.log(p2), dim=1, keepdim=True)), (-1, 1))
        entropy3 = torch.reshape(torch.sigmoid(-torch.sum(p3 * torch.log(p3), dim=1, keepdim=True)), (-1, 1))
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        # Total loss combines classification loss and entropy terms
        total_loss = loss + (1 - torch.mean(entropy2)) + torch.mean(entropy3)
        # Log the learning rate and other statistics
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(   'Session 0, epo {}, lrc={:.4f}, total loss={:.4f}, acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        # Update the average loss and accuracy
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        total_loss.backward()  # Only one backward pass here
        optimizer.step()
    # Return the average loss and accuracy for the epoch
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
def getRadius(trainset, transform, model, args):
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
    radius_list = [];
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_mean = embedding_this.mean(0)
        distances = torch.norm(embedding_this - embedding_mean, dim=1,p =2)
        #distances = torch.tensor(1) - F.linear(embedding_this, embedding_mean) #.squeeze()
        radius = torch.mean(distances)
        radius_list.append(radius)
    radius_list = torch.stack(radius_list, dim=0)
    print("radius",radius_list)
    return radius_list

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
def binaryClassifier_train(model,bc, indices_dict,dataBinay, session, args):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    bc = bc.train()
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss (for binary classification)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer (you can experiment with learning rate)
    embeddings ={}
    def hook_fn(module, input, output):
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(output.size(0), -1)
        embeddings['kk40'] = output

    def hook_fn1(module, input, output):
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(output.size(0), -1)
        embeddings['kk41'] = output

    if args.dataset == 'cifar100':
        relu_layer = model.encoder.layer3[0].relu
        relu_layer.register_forward_hook(hook_fn)
        relu_layer1 = model.encoder.layer3[1].relu
        relu_layer1.register_forward_hook(hook_fn1)
    else:
        relu_layer = model.encoder.layer4[0].relu
        relu_layer.register_forward_hook(hook_fn)
        relu_layer1 = model.encoder.layer4[1].relu
        relu_layer1.register_forward_hook(hook_fn1)
    epoch = 25
    for i in range(epoch):
        data, train_label = dataBinay["img"],dataBinay["label"].float().view(-1,1)  #[_.cpu() for _ in batch] #[_.cuda() for _ in batch]
        data =dataAug(data)
        embed = model.encode(data)
        #print(train_label)
        embed11 = get_activations_as_tensor(model, data, indices_dict)
        #embed11 = embed11 *torch.normal(1, 0.01, embed.shape) #,embeddings['kk41']
        #embed11 = torch.concat([embed,embeddings['kk40']],dim=1)
        ind = torch.randint(0, data.shape[0], (1, ))
        #print("activations:", data.shape,embed11.shape) #, embed11[ind,10:15])
        out = bc (embed11)
        loss = criterion(out, train_label)
        acc = count_acc1(out, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Session , binary loss={:.4f} acc={:.4f}'.format(session,loss, acc))
    return bc
def count_acc1(pred, labels):
    predicted = (pred > 0.5).float()  # Threshold probabilities for binary classification
    correct = (predicted == labels.view(-1, 1)).sum().item()  # View labels to match pred shape
    total = labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

def binaryClassifiedDataset(trainloader, binarylassifiedData, session,args):
    batch_size = trainloader.batch_size  # Get batch size
    with torch.no_grad():
        for i, (data, labels) in enumerate(trainloader):  # Iterate through the DataLoader correctly
            data = data.cpu()
            lab_tensor = torch.ones(batch_size).cpu()
            if session > 0:
                lab_tensor = torch.zeros(batch_size).cpu()  # Create zeros tensor for the batch
            # Select indices (if you still need to select a subset of images within the batch):
            ind = torch.randint(0, data.shape[0], (min(5, data.shape[0]),)) # Select up to 5 indices, or less if batch is smaller
            binarylassifiedData["img"] = torch.cat([binarylassifiedData["img"], data[ind]], dim=0)  # Use the whole batch
            binarylassifiedData["label"] = torch.cat([binarylassifiedData["label"], lab_tensor[ind]], dim=0)
    return binarylassifiedData

def get_low_weight_indices(model, percentage=0.1, num_layers=10):
    selected_layers = []
    for name, layer in reversed(list(model.named_modules())):  # Reverse order
        if hasattr(layer, 'weight') and hasattr(layer.weight, 'detach'):
            selected_layers.append(layer)
            if len(selected_layers) >= num_layers:
                break
    indices_dict = {}
    for layer in selected_layers:
        weight = layer.weight.detach().cpu().flatten()
        num_elements = weight.numel()
        k = max(1, int(num_elements * percentage))
        _, indices = torch.topk(torch.abs(weight), k, largest=False)
        indices_dict[layer] = indices  # Store indices
    return indices_dict

def get_activations_as_tensor(model, inputs, indices_dict):
    activation_store = {}
    def hook_fn(module, input, output):
        activation_store[module] = output.detach()
    hooks = []
    for layer in indices_dict.keys():
        hooks.append(layer.register_forward_hook(hook_fn))
    _ = model(inputs)
    batch_size = inputs.shape[0]
    activations_list = [[] for _ in range(batch_size)]
    #print(f"Stored activations keys: {list(activation_store.keys())}")
    for layer, indices in indices_dict.items():
        if layer in activation_store:
            activation_values = activation_store[layer].view(batch_size, -1)
            #print(f"Layer: {layer}, Activation Shape: {activation_values.shape}, Indices Shape: {indices.shape}")
            for i in range(batch_size):
                valid_indices = indices[indices < activation_values.shape[1]]
                if valid_indices.numel() > 0:
                    extracted_activations = activation_values[i, valid_indices]
                    #print(f"Extracted activations shape (Image {i}): {extracted_activations.shape}")
                    if extracted_activations.numel() > 0:
                        activations_list[i].append(extracted_activations)
    activations_tensors = []
    for i, activations in enumerate(activations_list):
        if activations:
            activations_tensors.append(torch.cat(activations, dim=0))
        else:
            #print(f"Warning: No activations found for image {i}")
            activations_tensors.append(torch.zeros(1))
    for hook in hooks:
        hook.remove()
    max_len = max(tensor.shape[0] for tensor in activations_tensors)
    activations_tensors = [torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in activations_tensors]
    final_tensor = torch.stack(activations_tensors)
    #print(f"Final activations tensor shape: {final_tensor.shape}")
    return final_tensor


def test1(model, model1, radii,fc2, testloader, bc, indices_dict,mcCentroid,epoch, args, session, result_list=None):
    test_class = args.base_class + session * args.way
    model = model.eval()  # Set model to evaluation mode
    model1 = model1.eval()
    bc = bc.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    # for name, module in model.encoder.named_modules():
    #     print(name)
    embeddings = {}
    def hook_fn(module, input, output):
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(output.size(0), -1)
        embeddings['kk40'] = output
    def hook_fn1(module, input, output):
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(output.size(0), -1)
        embeddings['kk41'] = output
    if args.dataset=='cifar100':
        relu_layer = model.encoder.layer3[0].relu
        relu_layer.register_forward_hook(hook_fn)
        relu_layer1 = model.encoder.layer3[1].relu
        relu_layer1.register_forward_hook(hook_fn1)
    else:
        relu_layer = model.encoder.layer4[0].relu
        relu_layer.register_forward_hook(hook_fn)
        relu_layer1 = model.encoder.layer4[1].relu
        relu_layer1.register_forward_hook(hook_fn1)
    rAvg = torch.mean(radii)
    # print(radii)
    #radii = torch.concat([radii, torch.ones(test_class - args.base_class) * rAvg])
    mw = model.fc.weight[:test_class,:]; mw1= model.fc1.weight[:test_class,:]; mw2= fc2.weight[:test_class,:]
    #mw2[:args.base_class,:] = torch.normal(0, 0.001, (args.base_class,mw2.shape[-1]))
    with (torch.no_grad()):
        for i, (data, test_label) in enumerate(testloader, 1):
            data, test_label = data.cpu(), test_label.cpu()  # Move data to CPU for compatibility
            #logits1, logits2 = model(data)
            embed = model.encode(data)
            # embed11 = embed + embeddings['kk40']*0.01
            # #embed12 = torch.concat([embed, embeddings['kk40']], dim=1)
            # embed12 = get_activations_as_tensor(model, data, indices_dict) #
            # #embed11 = innerFeature(model,embed, embeddings['kk40'],args)
            # bcPred = bc(embed12).squeeze()
            # embed11 = torch.where((bcPred.unsqueeze(1) >0.5), embed, embed11)
            logits1 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
            logits = logits1

            #logits ,b, cl = model1([logits1, logits2, logits3])
            #logitsMisclassified = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mcCentroid, p=2, dim=-1))* args.temperature
            #bcPred = bc(logits - logits1).squeeze()
            #bcPred = bc(torch.concat([logits1, logits2, logits3], dim=1)).squeeze()

            #print(session, bcPred.shape, bcPred[0], bcPred[-1], logits.shape, len(test_label))
            #pVar = torch.var(logits1, dim=1);  mxVar = torch.var(logits, dim=1); diffVar = torch.reshape(mxVar - pVar, (-1, 1))
            #print(pVar[0],pVar[40],diffVar[62])
            # logits4 = F.linear(F.normalize(embed11, p=2, dim=-1),F.normalize(mcCentroid, p=2, dim=-1)) * args.temperature
            #mx1 = torch.max(logits11,dim=1)
            # mx2 = torch.max(logits4,dim=1)[0]
            # #print("mcCentroid", mcCentroid.shape,logits1.shape,logits4.shape,mx1.shape,mx2.shape)
            #logits = torch.where((mx1[0].unsqueeze(1)> torch.mean(mx1[0]))& (mx1[1].unsqueeze(1)<args.base_class), logits11, logits) #mini
            #logits = torch.where(pVar.unsqueeze(1) < 0.8, logits1, logits)  # cifar
            #logits = torch.where(pVar.unsqueeze(1) < 1, logits1, logits)  # cub
            #logits = torch.where(diffVar < 0.1, logits1, logits)
            # #cond2 = (ind1> 60).unsqueeze(1)  & (max1 >0.5).unsqueeze(1)
            # #ind = torch.argmax(logits11, dim=1)
            #logits = torch.where(cond1, logits1,logits4)
            # if session == 8:
            #    writeFile(logits1, logits2,logits3, logits,i, test_label, model.fc.weight[:test_class,:])
            # #    writeFile(embed,  embeddings["kk40"], embeddings["kk41"], logits, i, test_label, model.fc.weight)
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
            #save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs)  # save_model_dir
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            p = np.sum(cm[:args.base_class, :test_class])
            fn = np.sum(cm[:args.base_class, args.base_class:test_class])
            n = np.sum(cm[args.base_class:test_class, :test_class])
            fp = np.sum(cm[args.base_class:test_class, :args.base_class])
            FP = np.round(np.mean(fp/n)*100,2) #- np.diag(cm)[args.base_class:])  # Sum of columns excluding diagonal
            FN = np.round(np.mean(fn/p)*100,2)
            # False positive and false negative ratios
            #FP_ratio =np.nan_to_num( FP / np.sum(cm, axis=0))  # Ratio relative to predictions
            #FN_ratio = np.nan_to_num(FN / np.sum(cm, axis=1)) # Ratio relative to ground truth
            print(cm.shape,FP,FN)
            result_list.append(f"Seen Acc:{seenac}  Unseen Acc:{unseenac} ")
            logging.info('epo {}, loss={:.4f} acc={:.4f}, acc@5={:.4f} '.format(epoch, vl, va, va5))
            return vl, (seenac, unseenac, va, FN,FP)
        else:
            return vl, va
def testEpisodic(model, model1, testloader, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()  # Set model to evaluation mode
    model1 = model1.eval(); #model2 = model2.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    embeddings = innerFeature(model,  args)
    mw = model.fc.weight.detach(); #model.fc.weight[:test_class,:].detach()
    mw1 =mw
    with (torch.no_grad()):
        for i, (data, test_label) in enumerate(testloader, 1):
            data, test_label = data.cpu(), test_label.cpu()  # Move data to CPU for compatibility (i < args.base_class): #
            if (session < 1) | (i not in list(range(test_class-args.way, test_class))):
                continue
            unique_batch_labels = test_label.unique().tolist()
            available_labels = list(range(args.base_class, test_class )) #torch.tensor(mw.shape[0])))
            remaining_labels = list(set(available_labels) - set(unique_batch_labels))
            needed = args.way - len(unique_batch_labels)
            extra_labels = random.sample(remaining_labels, needed)
            selected_classes = unique_batch_labels + extra_labels
            random.shuffle(selected_classes)  # Shuffle for randomness
            mw1 = mw[selected_classes]
            # Step 3: Create mapping from original label to new [0...args.way-1]
            label_map = {orig: new for new, orig in enumerate(selected_classes)}
            test_label = torch.tensor([label_map[l.item()] for l in test_label])
            #print(unique_batch_labels, available_labels,remaining_labels, selected_classes,test_label)
            embeddings.clear()
            H = model.encode(data).detach()
            z = embeddings['kk40']
            T = model1( [H,z]) #model1(H*z) #
            z0 = H*0.5 + T*0.5;
            #print(torch.std(z0)-torch.mean(z0))
            logits = F.linear(F.normalize(H, p=2, dim=-1), F.normalize(mw1, p=2, dim=-1)) * args.temperature

            acc = count_acc(logits, test_label)
            loss = F.cross_entropy(logits, test_label)
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
        if session > 0:
            return vl, (va5, va,va)
        else:
            return vl, va

def test2(model, fc2, testloader, epoch, args, session, result_list=None):
    test_class = args.base_class + session * args.way
    model = model.eval()  # Set model to evaluation mode
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    mw = model.fc.weight[:test_class, :]; mw1 = model.fc1.weight[:test_class, :];
    mw2 = fc2.weight[:test_class, :].clone()
    mw2[:args.base_class, :] = torch.normal(0, 0.001, (args.base_class, mw2.shape[-1]))
    with (torch.no_grad()):
        for i, (data, test_label) in enumerate(testloader, 1):
            data, test_label = data.cpu(), test_label.cpu()  # Move data to CPU for compatibility
            embed = model.encode(data)
            #logits = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(model.fc.weight[:test_class,:], p=2, dim=-1))* args.temperature
            logits1 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
            logits2 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw1, p=2, dim=-1)) * args.temperature
            logits3 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw2, p=2, dim=-1)) * args.temperature
            #logits = torch.where((test_label.unsqueeze(1) < args.base_class), logits2, logits3)
            # epsilon = 1e-10
            # p2 = torch.clamp(torch.nn.functional.softmax(logits2, dim=1), min=epsilon)
            # p3 = torch.clamp(torch.nn.functional.softmax(logits3, dim=1), min=epsilon)
            # entropy2 = torch.reshape(torch.sigmoid(-torch.sum(p2 * torch.log(p2), dim=1, keepdim=True)), (-1, 1))
            # entropy3 = torch.reshape(torch.sigmoid(-torch.sum(p3 * torch.log(p3), dim=1, keepdim=True)), (-1, 1))
            mx1,ind1 = torch.max(logits1,dim=1)
            cond1= (mx1.unsqueeze(1)>0.2) & (ind1.unsqueeze(1)<args.base_class)
            logits = torch.where(cond1, logits1, logits3)
            # cond2 = (mx1.unsqueeze(1) < 0.1) & (ind1.unsqueeze(1) < args.base_class)
            # logits = torch.where(cond1, logits1, torch.where(cond2, logits2,logits3))

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
            #save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs)  # save_model_dir
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
# Define the hook function
def print_layer_info(name, module, input, output):
    print(f"Layer: {name}")
    print(f"Output shape: {output.shape}")
# Register hooks to print the output dimension for each layer
def register_hooks(model):
    for name, layer in model.named_modules():
        # Register the hook for each layer that has a forward method
        if isinstance(layer, (nn.Conv2d, nn.ReLU)):#, nn.BatchNorm2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d)):
            layer.register_forward_hook(lambda module, input, output, name = name: print_layer_info(name, module, input, output))

from sklearn.cluster import KMeans
import torch
from sklearn.cluster import MiniBatchKMeans
def misClassifiedAndClustered(model, testloader, args, num_clusters=100):
    model.eval()  # Set the model to evaluation mode
    misclassified_ind = []  # Store misclassified indices
    misclassified_val = []  # Store misclassified confidence values
    misclassified_embeddings = []  # Store misclassified embeddings
    count = 0; cumulative_sum=[]
    with torch.no_grad():  # Disable gradient computations
        for i, batch in enumerate(testloader, 1):
            # Move data and labels to CPU
            data, test_label = [_.cpu() for _ in batch]
            embed = model.encode(data)
            logits = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(model.fc.weight[:args.base_class, :], p=2, dim=-1)) * args.temperature
            val, ind = torch.max(logits, dim=1)  # Get predicted confidence and indice
            # Identify misclassified indices with confidence below mean
            mn = torch.mean(val)
            misclassified_mask = (ind != test_label) & (val < mn)
            misclassified_indices = torch.arange(len(data))[misclassified_mask]
            misclassified_values = val[misclassified_mask]
            misclassified_embeds = embed[misclassified_mask]
            # Collect misclassified information
            misclassified_ind.append(misclassified_indices)
            misclassified_val.append(misclassified_values)
            misclassified_embeddings.append(misclassified_embeds)
            count += len(misclassified_indices)
            cumulative_sum.append(count)
    # Flatten embeddings into a single tensor
    if len(misclassified_embeddings) > 0:
        all_misclassified_embeddings = torch.cat(misclassified_embeddings, dim=0).cpu().numpy()
    else:
        print("No misclassified embeddings found.")
        return None, None
    # Perform KMeans clustering on the misclassified embeddings
    num_clusters = min(50, count // 10)  # If you don't need a tensor
    #kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #cluster_labels = kmeans.fit_predict(all_misclassified_embeddings) +args.base_class
    normalized_embeddings = all_misclassified_embeddings / np.linalg.norm(all_misclassified_embeddings, axis=1, keepdims=True)
    # Perform clustering using KMeans with cosine similarity (by using unit norm)
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=16)
    cluster_labels = kmeans.fit_predict(normalized_embeddings)

    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    print(f"Total Unique Misclassified Instances: {count}",misclassified_ind[5])
    print(f"Performed clustering into {num_clusters} clusters.",cluster_labels[cumulative_sum[4]:cumulative_sum[5]])
    return centroids #,misclassified_ind, misclassified_val,  cluster_labels


import torch.optim as optim
def session_train(model, model1,  fc2, radii, trainloader, optimizer,bc, epochs_new, scheduler,  session,args):
    model = model.train()
    model1 = model1.train()
    test_class = args.base_class + session * args.way
    #embed = model.fc.weight.data.clone().cpu()[:test_class, :]
    embed = model.fc.weight.data.detach().cpu()[:test_class, :]
    rAvg = torch.mean(radii)
    #radii =  torch.ones(test_class) * rAvg*0.7
    radii =  torch.concat([radii, torch.ones(test_class - args.base_class) * rAvg]) * 0.7
    epochs_new = 25 + session * args.way
    train_label = torch.arange(test_class)
    tqdm_gen = tqdm(trainloader)
    tqdm_gen_iter = iter(tqdm_gen)
    batch = next(tqdm_gen_iter)
    data1, train_label1 = [_.cpu() for _ in batch]  # Move to CPU if necessary
    train_label2 = torch.arange(test_class - args.way, test_class)
    #binary_label = torch.ones(test_class)*(train_label > args.base_class)
    # bc =bc.train()
    # criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    # optimizer1 = optim.Adam(model.parameters(), lr=0.001)
    mw = model.fc.weight[:test_class, :];    mw1 = model.fc1.weight[:test_class, :];   mw2 = fc2.weight[:test_class, :]
    for j in range(epochs_new):
        # data = dataAug(data1)
        # embed01 = model.encode(data);
        # embed0 = torch.mean(torch.reshape(embed01, (args.way,-1,embed01.shape[-1])), dim=1)
        #print(data.shape, embed0.shape,train_label1)
        #logits0 = F.linear(F.normalize(embed0, p=2, dim=-1), F.normalize(model.fc.weight[:test_class, :], p=2, dim=-1))#* args.temperature
        #logits0 = torch.cdist(embed0, model.fc.weight[:test_class, :], p=2).pow(2)
        #loss1 = F.cross_entropy(-logits0, train_label2)
        embed1 = embed #+ torch.normal(0.0, 0.001, embed.shape)

        random_values = torch.reshape(torch.rand(test_class),(-1,1))
        shuffled_row_indices = torch.randperm(embed.size(0))
        shuffled_tensor_2d = embed[shuffled_row_indices]
        embedFinal = (embed1+ F.normalize((shuffled_tensor_2d - embed1), p=2, dim=-1) * (torch.reshape(radii,(-1,1)) -0.01*random_values))
        logits1 = F.linear(F.normalize(embedFinal, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
        logits2 = F.linear(F.normalize(embedFinal, p=2, dim=-1), F.normalize(mw1, p=2, dim=-1)) * args.temperature
        logits3 = F.linear(F.normalize(embedFinal, p=2, dim=-1), F.normalize(mw2, p=2, dim=-1)) * args.temperature
        #logits =logits1 #
        logits, b, pred = model1 ([logits1,logits2,logits3])
        one_hot_labels = torch.nn.functional.one_hot(train_label, num_classes=test_class)
        ls1 = one_hot_labels*(logits-logits1)*(train_label >= args.base_class)
        constraint_penalty1 = F.relu( -torch.mean(ls1)) ** 2
        ls2 = one_hot_labels * (logits - logits1) * (train_label < args.base_class)
        constraint_penalty2 = F.relu(-1+torch.mean(ls2)) ** 2
        # bcPred = bc(logits - logits1).squeeze()
        # logits = torch.where((bcPred.unsqueeze(1) < 0.6), logits1 ,logits)
        acc = count_acc(logits[:, :test_class], train_label)
        loss = F.cross_entropy(logits[:, :test_class], train_label)
        total_loss = loss + constraint_penalty1 +  constraint_penalty2
        lrc = scheduler.get_last_lr()[0]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # loss1 = criterion(bcPred, binary_label)
        # optimizer1.zero_grad()
        # loss1.backward()
        # optimizer1.step()
        tqdm_gen.set_description('Session {}, total loss={:.4f} acc={:.4f} '.format(session, loss.item(), acc))
        #scheduler.step()
import torchvision.transforms as T
def dataAug(images, num_aug=5):
    augmentation_pipeline = T.Compose([
        T.RandomResizedCrop(size=(32,42), scale=(0.5, 1)),  # Slight zoom + random crop
        T.RandomHorizontalFlip(p=0.5), ])# 50% chance of flipping
    augmented_images = []
    for img in images:
        #for _ in range(num_aug):  # Apply augmentation `num_aug` times per image
        augmented_images.append(augmentation_pipeline(img))
    return torch.stack(augmented_images)

def updatedWeight(initial_weights, model):
    updated_weights = {k: v.clone() for k, v in model.state_dict().items()}  # Get a copy of current weights
    eta = 1;     alpha = 100
    # layer_names = list(initial_weights.keys())  # Get all layer names
    # penultimate_layer = layer_names[-2]  # Second-to-last layer
    new_state_dict = {}
    for (name, initial), (_, updated), layer_name in zip(initial_weights.items(), updated_weights.items(),initial_weights):
        weight_diff = updated - initial
        weight_decay_factor = torch.exp(-torch.abs(alpha * initial))
        modified_weight = initial + eta * weight_decay_factor *weight_diff #torch.abs( ) #)#
        new_state_dict[name] = modified_weight
        if layer_name =='fc.weight':
            new_state_dict[name] = updated
    model.load_state_dict(new_state_dict)
    return model
def session_train1(model, trainloader, optimizer, epochs_new, scheduler,  session,args):
    model = model.train()
    # for name, module in model.encoder.named_modules():
    #     print(name)
    test_class = args.base_class + session * args.way
    epochs_new = 25
    tqdm_gen = tqdm(trainloader)
    tqdm_gen_iter = iter(tqdm_gen)
    batch = next(tqdm_gen_iter)
    data1, train_label1 = [_.cpu() for _ in batch]  # Move to CPU if necessary
    train_label2 = torch.arange(test_class - args.way, test_class)
    initial_weights1 = {k: v.clone() for k, v in model.state_dict().items()}
    # for layer_name in initial_weights1:
    #     print(layer_name)
    wt = model.fc.weight[:test_class, :]; wt1 = model.fc1.weight[:test_class, :];#wt2 = fc2.weight[:test_class, :]
    for j in range(epochs_new):
        initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
        data = dataAug(data1)
        embed01 = model.encode(data)
        #embed01 = innerFeature(model, embed01, data1, args)
        #frac = torch.tensor([0, 0.2,-0.2,-0.3]); rv= frac[torch.randint(0, len(frac), (1,))]
        embed0 = torch.mean(torch.reshape(embed01, (-1,args.shot,embed01.shape[-1])), dim=1)
        embed0 = soft_calibration2(wt, embed0, args )
        #embed0 = wt[-args.way:, :] * rv + embed0 * torch.normal(1, 0.01, embed0.shape)
        #print(data1.shape, embed0.shape,train_label1,train_label2)
        #wt2 = wt * torch.normal(1, 0.01, wt.shape)
        logits = F.linear(F.normalize(embed0, p=2, dim=-1), F.normalize(wt, p=2, dim=-1))*args.temperature

        acc = count_acc(logits[:, :test_class], train_label2)
        loss = F.cross_entropy(logits[:, :test_class], train_label2)
        total_loss = loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        tqdm_gen.set_description('Session {}, total loss={:.4f} acc={:.4f} '.format(session, loss.item(), acc))
        scheduler.step()
        #if(j%5==3):
        model = updatedWeight(initial_weights, model)
        # cov_loss = covariance_loss(wt)  # Compute covariance penalty
        # #cov_loss = - mahalanobis_loss(wt)
        # alpha = 0.01  # Weight for covariance regularization
        # total_loss = alpha * cov_loss
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()
        #updated_weights = {k: v.clone() for k, v in model.state_dict().items()}
    #layer_name = model.encoder.layer3[0].relu
    #layer_name = "encoder.layer3.0.conv2.weight" #encoder.layer3[0].conv1"
    #transformed_filters = find_transformed_filters_with_weights(model,initial_weights1, updated_weights, layer_name)
    #feature_responses = compute_feature_response(model, data1, layer_name, transformed_filters)
    #print(feature_responses)
    return model
def soft_calibration2(base_protos,cur_protos, args):
    base_protos1 = F.normalize(base_protos, p=2, dim=-1)
    cur_protos1 = F.normalize(cur_protos, p=2, dim=-1)
    updated_protos = cur_protos1
    itr = torch.randint(0,10,(1,))
    for i in range(itr):
        softmax_t = torch.randint(1, 4, (1,)) * 4
        weights = torch.mm(updated_protos, base_protos1.T) * softmax_t #args.softmax_t
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        updated_protos = (1 - args.shift_weight) * updated_protos + args.shift_weight * delta_protos
    return updated_protos

def covariance_loss(fc_weights):
    #Minimizes the off-diagonal elements of the covariance matrix of the prototype weights.
    # Compute mean along feature dimension
    W = fc_weights  # Shape: (num_classes, feature_dim)
    W_mean = W.mean(dim=0, keepdim=True)  # Shape: (1, feature_dim)
    # Center the weights (zero mean)
    W_centered = W - W_mean  # Shape: (num_classes, feature_dim)
    # Compute covariance matrix
    C = (W_centered.T @ W_centered) / W.shape[0]  # Shape: (feature_dim, feature_dim)
    # Extract off-diagonal elements
    off_diagonal = C - torch.diag(torch.diag(C))
    # Compute loss as squared sum of off-diagonal elements
    cov_loss = torch.sum(off_diagonal ** 2)
    return cov_loss

import csv
def writeFile(logits1, logits2, logits3, logits,i, test_label, proto):
    epsilon = 1e-10
    selected_logits1, selected_logits2, selected_logits3, selected_logits , index = [], [], [], [],[]
    unique_labels = torch.unique(test_label)
    #print(i,test_label)
    for label in unique_labels:
        indices = torch.where(test_label == label)[0]
        if len(indices) >= 4:  # Ensure at least 2 samples per label
            selected_indices = indices[:4]  # Select first two instances
            selected_logits1.append(logits1[selected_indices])
            selected_logits2.append(logits2[selected_indices])
            selected_logits3.append(logits3[selected_indices])
            selected_logits.append(logits[selected_indices])
            #index.append(torch.tensor([label, label], dtype=torch.long).unsqueeze(1))

    if selected_logits1:  # Ensure non-empty selection before concatenation
        selected_logits1 = torch.cat(selected_logits1, dim=0)
        selected_logits2 = torch.cat(selected_logits2, dim=0)
        selected_logits3 = torch.cat(selected_logits3, dim=0)
        selected_logits = torch.cat(selected_logits, dim=0)
        #index = torch.cat(index, dim=0)
        result_tensor = torch.cat([selected_logits1, selected_logits2, selected_logits3, selected_logits], dim=1)
        result_array = result_tensor.detach().numpy()
        # Determine file mode
        file_mode = "w" if i == 0 else "a"
        # with open("logits_output_cifar.csv", mode=file_mode, newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows(result_array)
        with open("logit123_output_cifar_.csv", mode=file_mode, newline="") as file:
            writer = csv.writer(file)
            writer.writerows(result_array)
        result_array = proto.detach().numpy()
        print(proto.shape,result_array.shape)
        # with open("proto_cifar.csv", mode=file_mode, newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows(result_array)
        # with open("logit_nearViewDiff_mini.csv", mode=file_mode, newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows(result_array)

def attention(query, key, value, mask=None, dropout=None):
    query = query.unsqueeze(1)  # Shape: (batch_size, 1, d_k)
    key = key.unsqueeze(1)  # Shape: (batch_size, 1, d_k)
    value = value.unsqueeze(1)  # Shape: (batch_size, 1, d_k)
    batch_size, seq_len, d_k = query.size() #batch_size, seq_len, d_k
    # Compute dot-product attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # Shape: [batch_size, seq_len, seq_len]
    scores = scores - scores.max(dim=-1, keepdim=True)[0]  # Normalize for numerical stability
    # Normalize scores using softmax
    attention_weights = F.softmax(scores, dim=-1)  # Shape: [batch_size, seq_len, seq_len]
     # Compute the attention output
    output = torch.matmul(attention_weights, value)  # Shape: [batch_size, seq_len, d_v]
    output = torch.mean(output, dim=1)
    return output, attention_weights


def find_transformed_filters_with_weights(model,initial_weights, updated_weights, layer_name, lowest_k=100, top_k=10):
    if layer_name not in initial_weights or layer_name not in updated_weights:
        raise ValueError(f"Layer '{layer_name}' not found in model parameters.")
    # Get filter weights (sum absolute values per filter)
    initial_layer_weights = initial_weights[layer_name].view(initial_weights[layer_name].shape[0], -1).sum(dim=1)
    updated_layer_weights = updated_weights[layer_name].view(updated_weights[layer_name].shape[0], -1).sum(dim=1)
    # Find indices of lowest 100 filters in initial weights
    lowest_indices = torch.argsort(initial_layer_weights)[:lowest_k]
    # Find indices of top 10 filters in updated weights
    top_indices = torch.argsort(updated_layer_weights, descending=True)[:top_k]
    # Find common indices
    selected_indices = set(lowest_indices.tolist()) & set(top_indices.tolist())
    # Store index with initial and updated weights
    transformed_filters = [(idx, initial_layer_weights[idx].item(), updated_layer_weights[idx].item()) for idx in
                           selected_indices]
    idxx =[]
    for idx, init_weight, upd_weight in transformed_filters:
        idxx.append(idx)
        #print(f"Filter {idx}: Initial weight = {init_weight:.10f}, Updated weight = {upd_weight:.10f}")
    # print(idxx)
    # image = torch.randn(3, 32, 32)  # Example image (assuming 3-channel, 32x32)
    # feature_responses = compute_feature_response(model, image, layer_name, transformed_filters)
    # print(feature_responses)
    return transformed_filters

def compute_feature_response(model, images, layer_name, transformed_filters):
    layer_name = "layer3.0.conv1" # model.encoder.layer3[0].conv2
    #layer_name = "encoder.layer3.0.conv2.weight"
    feature_responses = {}
    # Hook function to extract feature map
    def hook_fn(module, input, output):
        nonlocal feature_map
        feature_map = output  # Store the output feature map
    # Register hook at the target layer
    feature_map = None
    for name, module in model.encoder.named_modules():
        #print(name)
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    else:
        raise ValueError(f"Layer '{layer_name}' not found in the model.")
    # Forward pass to get feature maps
    with torch.no_grad():
        _ = model(images)  # Forward pass on the batch
    handle.remove()
    total_sum =torch.zeros(images.shape[0], device=images.device)
    for idx, _, _ in transformed_filters:
        activation = feature_map[:, idx].sum(dim=[1, 2])  # Shape: (B,)
        total_sum += activation
    return total_sum





