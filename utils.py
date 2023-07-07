import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from meta import *
from pre_resnet10 import PreActResNet18
import os



def adv_feature(logits_vector,loss_vector,labels,loss_tensor_last, class_rate):
    # feature 2 batch mean classes
    labels_one_hot = F.one_hot(labels,num_classes=10).float()
    class_rate = torch.mm(labels_one_hot,class_rate.unsqueeze(1))
    last_epoch_ave_loss = torch.mm(labels_one_hot,loss_tensor_last.unsqueeze(1))
    # feature 4 preds
    logits_labels = torch.sum(F.softmax(logits_vector,dim=1) * labels_one_hot,dim=1)
    # feature 5 grad
    logits_vector_grad = torch.norm((labels_one_hot- F.softmax(logits_vector,dim=1)),dim=1)    
    # feature 6 margin
    logits_others_max =(F.softmax(logits_vector,dim=1)[labels_one_hot!=1].reshape(F.softmax(logits_vector,dim=1).size(0),-1)).max(dim=1).values
    logits_margin =  logits_labels - logits_others_max
    entropy =  torch.sum(F.softmax(logits_vector,dim=1)*F.log_softmax(logits_vector,dim=1),dim=1)
    
    feature = torch.cat([loss_vector.unsqueeze(1),
                        last_epoch_ave_loss,
                        logits_vector_grad.unsqueeze(1),
                        logits_margin.unsqueeze(1),
                        entropy.unsqueeze(1),
                        class_rate],dim=1)
    return feature

def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta  
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd


def pgd_attack_frl(model,
                  X,
                  y,
                  weight,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    new_eps = (epsilon * weight).view(weight.shape[0], 1, 1, 1)  

    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        eta = torch.min(torch.max(X_pgd - X.data, -1.0 * new_eps), new_eps)
        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd

def in_class(predict, label):

    probs = torch.zeros(10)
    for i in range(10):
        in_class_id = torch.tensor(label == i, dtype= torch.float)
        correct_predict = torch.tensor(predict == label, dtype= torch.float)
        in_class_correct_predict = (correct_predict) * (in_class_id)
        acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
        probs[i] = acc

    return probs

def match_weight(label, diff0, diff1, diff2): 

    weight0 = torch.zeros(label.shape[0], device='cuda')
    weight1 = torch.zeros(label.shape[0], device='cuda')
    weight2 = torch.zeros(label.shape[0], device='cuda')

    for i in range(10):
        weight0 += diff0[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight1 += diff1[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight2 += diff2[i] * torch.tensor(label == i, dtype= torch.float).cuda()

    return weight0, weight1, weight2



def cost_sensitive(lam0, lam1, lam2):

    ll0 = torch.clone(lam0)
    ll1 = torch.clone(lam1)

    diff0 = torch.ones(10) * 1 / 10
    for i in range(10):
        for j in range(10):
            if j == i:
                diff0[i] = diff0[i] + 9 / 10 * ll0[i]
            else:
                diff0[i] = diff0[i] - 1 / 10 * ll0[j]

    diff1 = torch.ones(10) * 1/ 10
    for i in range(10):
        for j in range(10):
            if j == i:
                diff1[i] = diff1[i] + 9 / 10 * ll1[i]
            else:
                diff1[i] = diff1[i] - 1 / 10 * ll1[j]
    diff2 = torch.clamp(torch.exp(2 * lam2), min = 0.98, max = 2.5)

    return diff0, diff1, diff2


def evaluate(model, test_loader, configs, device, mode = 'Test'):
    
    print('Doing evaluation mode ' + mode)
    model.eval()

    correct = 0
    correct_adv = 0

    all_label = []
    all_pred = []
    all_pred_adv = []

    for batch_idx, (index, data, target) in enumerate(test_loader):

        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        all_label.append(target)

        ## clean test
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) 
        add = pred.eq(target.view_as(pred)).sum().item()
        correct += add
        model.zero_grad()
        all_pred.append(pred)

        ## adv test
        x_adv = pgd_attack(model, X = data, y = target, **configs)
        output1 = model(x_adv)
        pred1 = output1.argmax(dim=1, keepdim=True)  
        add1 = pred1.eq(target.view_as(pred1)).sum().item()
        correct_adv += add1
        all_pred_adv.append(pred1)

    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()

    acc = in_class(all_pred, all_label)
    acc_adv = in_class(all_pred_adv, all_label)

    total_clean_error = 1- correct / len(test_loader.dataset)
    total_bndy_error = correct / len(test_loader.dataset) - correct_adv / len(test_loader.dataset)
    total_error = 1 - correct_adv / len(test_loader.dataset)
    

    class_clean_error = 1 - acc
    class_bndy_error = acc - acc_adv 
    class_total_error = 1-acc_adv
    

    return correct_adv / len(test_loader.dataset), class_clean_error, class_bndy_error, class_total_error, total_clean_error, total_bndy_error, total_error




def trades_adv(model,
               x_natural,
               weight,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    # define KL-loss
    new_eps = (epsilon * weight).view(weight.shape[0], 1, 1, 1)

    criterion_kl = nn.KLDivLoss(size_average = False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - new_eps), x_natural + new_eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def caat_adv(model,
               x_natural,
               y_natural,
               weight,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    # define KL-loss
    # print("epsilon")
    # print(epsilon)
    new_eps = (epsilon * weight).view(weight.shape[0], 1, 1, 1)

    criterion_kl = nn.KLDivLoss(size_average = False)
    criterion_ce = nn.CrossEntropyLoss()
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_anti_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach() 
    for _ in range(num_steps):
        x_adv.requires_grad_()
        x_anti_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
            anti_loss = criterion_ce(F.log_softmax(model(x_anti_adv), dim=1),
                                y_natural)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        anti_grad = torch.autograd.grad(anti_loss, [x_anti_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_anti_adv = x_anti_adv.detach() - step_size * torch.sign(anti_grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - new_eps), x_natural + new_eps)
        x_anti_adv = torch.min(torch.max(x_anti_adv, x_natural - new_eps), x_natural + new_eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_anti_adv = torch.clamp(x_anti_adv, 0.0, 1.0)
    return x_adv, x_anti_adv


def train_model(class_rate, loss_tensor_last, lr, model, meta_net, train_loader, valid_loader, meta_optimizer, optimizer, diff0, diff1, diff2, epoch, beta, configs, device):

    criterion_kl = nn.KLDivLoss(reduction='none')
    criterion_nat = nn.CrossEntropyLoss(reduction='none')
    criterion_meta = nn.CrossEntropyLoss()
    meta_dataloader_iter = iter(valid_loader)
    print('Doing Training on epoch:  ' + str(epoch))
    model.train()
    class_total = torch.zeros([10]).cuda()
    loss_total = torch.zeros([10]).cuda()
    for batch_idx, (_, data, target) in enumerate(train_loader):
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        labels_one_hot = F.one_hot(target,num_classes=10).float()
        weight0, weight1, weight2 = match_weight(target, diff0, diff1, diff2)
        if (batch_idx + 1) % 50 == 0:
            pseudo_model = PreActResNet18().to(device)
            pseudo_model.load_state_dict(model.state_dict())
            pseudo_model.train()
            ## get loss
            pseudo_logits = pseudo_model(data)
            pseudo_logits_labels = torch.sum(F.softmax(pseudo_logits,dim=1) * labels_one_hot,dim=1)
            pseudo_loss_natural = criterion_nat(pseudo_logits, target)
            
            #get weight
            pseudo_feature = adv_feature(pseudo_logits,pseudo_loss_natural,target,loss_tensor_last, class_rate)
            pseudo_weight = meta_net(pseudo_feature)
            ## generate adv examples
            x_adv, x_anti_adv = caat_adv(pseudo_model, x_natural = data, y_natural = target, weight = weight2, **configs) #这个weight2就是控制阈值的那个数字
            pseudo_anti_logits = pseudo_model(x_anti_adv)
            pseudo_loss_bndy_vec = criterion_kl(F.log_softmax(pseudo_model(x_adv), dim=1), F.softmax(pseudo_model(data), dim=1))
            pseudo_loss_bndy = torch.sum(pseudo_loss_bndy_vec, 1)
            pseudo_loss_anti = criterion_nat(pseudo_anti_logits, target)
            pseudo_loss = (torch.sum(pseudo_weight[:,0] * pseudo_loss_natural * weight0)/ torch.sum(weight0)\
                + beta * torch.sum(pseudo_weight[:,0] * pseudo_loss_bndy * weight1) / torch.sum(weight1)) + torch.mean(pseudo_weight[:,1]* pseudo_loss_anti)
            
            pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_model.parameters(), create_graph=True)   
            pseudo_optimizer = MetaSGD(pseudo_model, pseudo_model.parameters(), lr=lr)
            pseudo_optimizer.load_state_dict(optimizer.state_dict())
            pseudo_optimizer.meta_step(pseudo_grads)
            del pseudo_grads

            try:
                _, meta_inputs, meta_labels = next(meta_dataloader_iter)
            except StopIteration:
                meta_dataloader_iter = iter(valid_loader)
                _, meta_inputs, meta_labels = next(meta_dataloader_iter)

            meta_inputs, meta_labels = meta_inputs.cuda(), meta_labels.cuda()
            meta_labels_one_hot = F.one_hot(meta_labels,num_classes=10).float()
            meta_outputs = pseudo_model(meta_inputs)
            meta_logits_labels = torch.sum(F.softmax(meta_outputs,dim=1) * meta_labels_one_hot,dim=1)
            meta_loss_nat = criterion_nat(meta_outputs, meta_labels.long()) # dim1
            meta_x_adv, meta_x_anti_adv = caat_adv(pseudo_model, x_natural = meta_inputs, y_natural = meta_labels, weight = weight2, **configs)
            meta_anti_logits = pseudo_model(meta_x_anti_adv)
            meta_loss_bndy_vec = criterion_kl(F.log_softmax(pseudo_model(meta_x_adv), dim=1), F.softmax(pseudo_model(meta_inputs), dim=1))
            meta_loss_bndy = torch.sum(meta_loss_bndy_vec, 1)
            meta_loss_anti = criterion_meta(meta_anti_logits, meta_labels) #dim 1
            meta_loss = (torch.sum(meta_loss_nat * weight0)/ torch.sum(weight0)\
               + beta * torch.sum(meta_loss_bndy * weight1) / torch.sum(weight1)) + meta_loss_anti
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
        
        logits = model(data)
        logits_labels = torch.sum(F.softmax(logits,dim=1) * labels_one_hot,dim=1)
        loss_natural = criterion_nat(logits, target) 
        x_adv, x_anti_adv = caat_adv(model, x_natural = data, y_natural = target, weight = weight2, **configs)
        anti_logits = model(x_anti_adv) 
        loss_bndy_vec = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(data), dim=1))# torch.Size([128, 10])
        loss_bndy = torch.sum(loss_bndy_vec, 1) #torch.Size([128])
        loss_natural_anti = criterion_nat(anti_logits, target) 
        # loss_bndy_anti = torch.sum(loss_bndy_vec_anti, 1)

        with torch.no_grad():
            feature = adv_feature(logits,loss_natural,target,loss_tensor_last,class_rate)
            weights = meta_net(feature)
        
        loss = (torch.sum(weights[:,0] * loss_natural * weight0)/ torch.sum(weight0)\
               + beta * torch.sum(weights[:,0] * loss_bndy * weight1) / torch.sum(weight1)) + torch.mean(weights[:,1]*loss_natural_anti)
        labels_one_hot = F.one_hot(target,num_classes=10).float()
        class_total += torch.sum(labels_one_hot,dim=0)
        loss_total += torch.mm(loss_natural.detach().unsqueeze(0),labels_one_hot)[0]   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_tensor_last = loss_total/(class_total + 1e-6)
        ## clear grads
        



def frl_train(class_rate, loss_tensor_last, lr, h_net, ds_train, ds_valid, ds_test, meta_net, meta_optimizer, optimizer, now_epoch, configs, configs1, device, delta0, delta1, rate1, rate2, lmbda, beta, lim):
    print('train epoch ' + str(now_epoch), flush=True)
    best_acc = 0.
    ## given model, get the validation performance and gamma
    _, class_clean_error, class_bndy_error, _, total_clean_error, total_bndy_error, _ = \
        evaluate(h_net, ds_valid, configs1, device, mode='Validation')

    ## get gamma on validation set
    gamma0 = class_clean_error - total_clean_error - delta0
    gamma1 = class_bndy_error - total_bndy_error - delta1

    ## print inequality results
    print('total clean error ' + str(total_clean_error))
    print('total boundary error ' + str(total_bndy_error))

    print('.............')
    print('each class inequality constraints')
    print(gamma0)
    print(gamma1)

    lmbda0 = lmbda[0:10] + rate1 * torch.clamp(gamma0, min = -1000)
    lmbda1 = lmbda[10:20] + rate1 * 2 * torch.clamp(gamma1, min = -1000)
    lmbda2 = lmbda[20:30] + rate2 * gamma1

    lmbda0 = normalize_lambda(lmbda0, lim)
    lmbda1 = normalize_lambda(lmbda1, lim)  

    lmbda = torch.cat([lmbda0, lmbda1, lmbda2])
    diff0, diff1, diff2 = cost_sensitive(lmbda0, lmbda1, lmbda2)

    print('..............................')
    print('current lambda after update')
    print(lmbda0)
    print(lmbda1)
    print(lmbda2)

    print('..............................')
    print('current weight')
    print(diff0)
    print(diff1)
    print(diff2)
    print('..............................')
    _ = train_model(class_rate, loss_tensor_last, lr, h_net, meta_net, ds_train, ds_valid, meta_optimizer, optimizer, diff0, diff1, diff2, now_epoch,
                    beta, configs, device)
    test_robust_acc, test_class_clean_error, test_class_bndy_error, test_class_total_error, test_total_clean_error, test_total_bndy_error, test_total_error = \
        evaluate(h_net, ds_test, configs1, device, mode='test')
    
    print("test_class_clean_error")
    print(test_class_clean_error)
    print("test_class_bndy_error")
    print(test_class_bndy_error)
    print("test_class_total_error")
    print(test_class_total_error)
    print('Epoch: {}, (total_clean_error, total_bndy_error, total_error) Test: ({:.4f}, {:.2%}, {:.2%}) LR: {}'.format(
            now_epoch,
            test_total_clean_error,
            test_total_bndy_error,
            test_total_error,
            lr,
        ))
    

    return lmbda

def normalize_lambda(lmb, lim = 0.8):

    lmb = torch.clamp(lmb, min=0)
    if torch.sum(lmb) > lim:
        lmb = lim * lmb / torch.sum(lmb)
    else:
        lmb = lmb
    return lmb
