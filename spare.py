def session_train(model, model1,  fc2, radii, trainloader, optimizer, epochs_new, scheduler,  session,args):
    model = model.train()
    model1 = model1.train()
    test_class = args.base_class + session * args.way
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
    tqdm_gen = tqdm(trainloader)
    embed = model.fc.weight.data.detach().cpu()[:test_class,:]

    radii = torch.concat([radii, torch.ones(test_class - args.base_class) * 0.25])
    for j in range(epochs_new):
        k=50
        baseInd = torch.randint(0, args.base_class, (k,))
        newInd = torch.arange(args.base_class, test_class)
        indices = torch.cat((baseInd, newInd))
        embed1 = embed[indices] #+ torch.normal(0.0, 0.0001, embed.shape)
        raddi = radii[indices]
        list_tensor = torch.randint(0, test_class, (test_class - args.base_class + k,)) #torch.arange(test_class)
        list_tensor = list_tensor[torch.randperm(list_tensor.size(0))]
        embed2 = embed[list_tensor]
        random_values = torch.rand(test_class - args.base_class + k)
        #print(len(random_values), len(radii), len(embed1),len(embed2))
        embedFinal = embed1 + F.normalize((embed2 - embed1), p=2, dim=-1) * torch.reshape(raddi * random_values,(-1,1))
        train_label = indices #torch.arange(test_class)

        logits1 = F.linear(F.normalize(embedFinal, p=2, dim=-1), F.normalize(model.fc.weight, p=2, dim=-1))
        logits2 = F.linear(F.normalize(embedFinal, p=2, dim=-1), F.normalize(model.fc1.weight, p=2, dim=-1))
        logits3 = F.linear(F.normalize(embedFinal, p=2, dim=-1), F.normalize(fc2.weight, p=2, dim=-1))

        logits1 = logits1[:, :test_class] * args.temperature
        logits2 = logits2[:, :test_class] * args.temperature
        logits3 = logits3[:, :test_class] * args.temperature

        logits,b, entropy = model1 ([logits1,logits2,logits3])

        acc = count_acc(logits, train_label)
        loss = F.cross_entropy(logits, train_label)

        total_loss = loss
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description('Session {}, lrc={:.4f},total loss={:.4f} acc={:.4f} '.format(session, lrc, total_loss.item(), acc))

        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer1.step()

def session_train1(model,  fc2, trainloader, optimizer, epochs_new, scheduler, session, args):
        model = model.train()
        torch.autograd.set_detect_anomaly(True)
        test_class = args.base_class + session * args.way
        tqdm_gen = tqdm(trainloader)
        #embed = model.fc.weight.data.detach().cpu()[test_class-args.way:test_class, :]
        embed = model.fc.weight.data.detach().cpu()[:test_class, :]
        # print(embed.shape, epochs_new,torch.arange(test_class-args.way,test_class))
        for j in range(epochs_new):
            embed1 = embed + torch.normal(0.0, 0.1, embed.shape)
            train_label = torch.arange(0,test_class)
            #train_label = torch.arange(test_class - args.way, test_class)

            logits1 = F.linear(F.normalize(embed1, p=2, dim=-1), F.normalize(model.fc.weight, p=2, dim=-1))
            logits1 = logits1[:, :test_class] * args.temperature

            logits2 = F.linear(F.normalize(embed1, p=2, dim=-1), F.normalize(model.fc1.weight, p=2, dim=-1))
            logits2 = logits2[:, :test_class] * args.temperature

            logits3 = F.linear(F.normalize(embed1, p=2, dim=-1), F.normalize(fc2.weight, p=2, dim=-1))
            logits3 = logits3[:, :test_class] * args.temperature
            #logits = (logits1 + logits2)/2 #+logits3)/3 #torch.where(torch.argmax(logits1, dim=1).unsqueeze(1) == torch.argmax(logits2, dim=1).unsqueeze(1),  logits1, logits3)
            epsilon = 1e-10
            p1 = torch.clamp(torch.nn.functional.softmax(logits1, dim=1), min=epsilon)
            p2 = torch.clamp(torch.nn.functional.softmax(logits2, dim=1), min=epsilon)
            p3 = torch.clamp(torch.nn.functional.softmax(logits3, dim=1), min=epsilon)

            # Compute entropy for each
            entropy1 = -torch.sum(p1 * torch.log(p1), dim=1, keepdim=True)  # Shape: [batch_size, 1]
            entropy2 = -torch.sum(p2 * torch.log(p2), dim=1, keepdim=True)  # Shape: [batch_size, 1]
            entropy3 = -torch.sum(p3 * torch.log(p3), dim=1, keepdim=True)  # Shape: [batch_size, 1]
            beta = 0.2;  gamma = 0.1
            # logits = torch.where(pred1.unsqueeze(1) == pred2.unsqueeze(1), logits2, logits3)
            # logits = torch.where(pred1.values.unsqueeze(1) > pred3.values.unsqueeze(1), logits1, logits3)
            # logits = beta*logits1 * entropy1 + (1-beta)*(logits3 * entropy3 - logits2 * entropy2*gamma)
            logits = logits1 * entropy1 + torch.abs(beta*logits3 * entropy3 - logits2 * entropy2*gamma)


            acc = count_acc(logits, train_label)
            loss = F.cross_entropy(logits, train_label)

            total_loss = loss
            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(  'Session {}, lrc={:.4f},total loss={:.4f} acc={:.4f} '.format(session, lrc, total_loss.item(), acc))
            # if j%4==0:
            #     print(f"Epoch {j }, Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
