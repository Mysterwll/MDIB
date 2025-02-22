import sys

import torch
import torch.utils.data
from tqdm import tqdm


"""
run network Fusion_radio_img
"""
def run(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        running_loss = test_loss = 0.0
        for i, (radio, img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            radio = radio.to(device)
            img = img.to(device)
            label = label.to(device)
            outputs, IB_loss, disen_loss, task_loss = model(radio, img, label)
            loss = criterion(IB_loss, disen_loss, task_loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        observer.log(f"Train Loss: {train_loss:.4f}\n")
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (radio, img, label) in enumerate(test_bar):
                radio = radio.to(device)
                img = img.to(device)
                label = label.to(device)
                outputs, _ = model(radio, img, None)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(probabilities, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]
                observer.update(predictions, label, confidence_scores)
                test_loss += loss.item() * img.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            observer.log(f"Test Loss: {test_loss:.4f}\n")
        observer.record_loss(epoch, train_loss, test_loss)
        if observer.excute(epoch, model):       
            print("Early stopping")
            break
    observer.finish()


def run_0(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        running_loss = test_loss = 0.0
        for i, (radio, img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            radio = radio.to(device)
            img = img.to(device)
            label = label.to(device)
            outputs, logit = model(radio, img)
            loss = criterion(outputs, logit, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        observer.log(f"Train Loss: {train_loss:.4f}\n")
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (radio, img, label) in enumerate(test_bar):
                radio = radio.to(device)
                img = img.to(device)
                label = label.to(device)
                outputs = model(radio, img)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(probabilities, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]
                observer.update(predictions, label, confidence_scores)
                test_loss += loss.item() * img.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            observer.log(f"Test Loss: {test_loss:.4f}\n")
        observer.record_loss(epoch, train_loss, test_loss)
        if observer.excute(epoch, model):       
            print("Early stopping")
            break
    observer.finish()

def run_all(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        running_loss = test_loss = 0.0
        for i, (cli, radio, img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            cli, radio, img, label = cli.to(device), radio.to(device), img.to(device), label.to(device)
            outputs, IB_loss, disen_loss, diff_loss, task_loss = model(cli, radio, img, label)
            loss = criterion(IB_loss, disen_loss, diff_loss, task_loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        observer.log(f"Train Loss: {train_loss:.4f}\n")
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (cli, radio, img, label) in enumerate(train_bar):
                cli, radio, img, label = cli.to(device), radio.to(device), img.to(device), label.to(device)
                outputs = model(cli, radio, img, None)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(probabilities, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]
                observer.update(predictions, label, confidence_scores)
                test_loss += loss.item() * img.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            observer.log(f"Test Loss: {test_loss:.4f}\n")
        observer.record_loss(epoch, train_loss, test_loss)
        if observer.excute(epoch, model):       
            print("Early stopping")
            break
    observer.finish()

def run_img(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for i, (img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (img, label) in enumerate(test_bar):
                img = img.to(device)
                label = label.to(device)
                outputs = model(img)

                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(probabilities, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]

                observer.update(predictions, label, confidence_scores)
        if observer.excute(epoch, model):       
            print("Early stopping")
            break
    observer.finish()


def run_cmib(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for i, (cli, radio, img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            cli, radio, img, label = cli.to(device), radio.to(device), img.to(device), label.to(device)
            
            outputs, loss = model(img, radio, cli, label)
            # loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (cli, radio, img, label) in enumerate(test_bar):
                cli, radio, img, label = cli.to(device), radio.to(device), img.to(device), label.to(device)
                outputs = model.test(img, radio, cli)

                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(probabilities, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]

                observer.update(predictions, label, confidence_scores)
        if observer.excute(epoch, model):       
            print("Early stopping")
            break
    observer.finish()

def run_mpgsurv(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for i, (cli, radio, img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            cli, radio, img, label = cli.to(device), radio.to(device), img.to(device), label.to(device)
            
            outputs = model(img, radio, cli)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (cli, radio, img, label) in enumerate(test_bar):
                cli, radio, img, label = cli.to(device), radio.to(device), img.to(device), label.to(device)
                outputs = model(img, radio, cli)

                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(probabilities, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]

                observer.update(predictions, label, confidence_scores)
        if observer.excute(epoch, model):       
            print("Early stopping")
            break
    observer.finish()


def training_loop_contrastive(observer, epochs, train_loader, val_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        running_loss = val_loss = 0.0
        for i, (batch_data) in enumerate(train_bar):
            optimizer.zero_grad()
            if isinstance(batch_data, dict):
                inputs, inputs_2 = (
                        batch_data["image"].to(device),
                        batch_data["image_2"].to(device),
                    )
            else:
                inputs, inputs_2 = batch_data[0].to(device), batch_data[1].to(device)

            _, outputs= model(inputs)
            _, outputs_2= model(inputs_2)
            loss, _, _, _ = criterion(outputs, outputs_2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        with torch.no_grad():
            model.eval()
            val_bar = tqdm(val_loader, leave=True, file=sys.stdout)

            for i, (batch_data) in enumerate(val_bar):
                if isinstance(batch_data, dict):
                    inputs, inputs_2 = (
                            batch_data["image"].to(device),
                            batch_data["image_2"].to(device),
                        )
                else:
                    inputs, inputs_2 = batch_data[0].to(device), batch_data[1].to(device)
                _, outputs= model(inputs)
                _, outputs_2= model(inputs_2)
                loss, _, _, _ = criterion(outputs, outputs_2)
                val_loss += loss.item() * inputs.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            
        observer.record(epoch, model, train_loss, val_loss)

        # if observer.excute(epoch, model):       
        #     print("Early stopping")
        #     break
    observer.finish()

# def train_gcn(observer, g, idx_train, idx_val, epochs, model, device, optimizer, criterion):
#     print("start training")

#     model = model.to(device) 
#     features = g.ndata['h'].to(device) 
#     e_feature = g.edata['w'].to(device) 
#     labels = g.ndata['label'].to(device) 
#     g = g.to(device) 
    
#     for epoch in range(epochs):
#         print(f"Epoch: {epoch + 1}/{epochs}")

#         observer.reset()
#         model.train()
#         optimizer.zero_grad()
#         output = model(g, features,e_feature)
#         # Compute loss
#         # Note that you should only compute the losses of the nodes in the training set.
#         loss_train = criterion(output[idx_train], labels[idx_train])
        
#         loss_train.backward(retain_graph=True)
#         optimizer.step()
#         loss_train = loss_train.item()

#         with torch.no_grad():
#             model.eval()
#             val_output = model(g, features,e_feature)
#             loss_val = criterion(val_output[idx_val], labels[idx_val])

#             outputs = val_output[idx_val]

#             probabilities = torch.softmax(outputs, dim=1)
#             _, predictions = torch.max(probabilities, dim=1)
#             confidence_scores = probabilities[range(len(predictions)), 1]
#             observer.update(predictions, labels, confidence_scores)
#             test_loss = loss_val.item() 
#         observer.log(f"Test Loss: {test_loss:.4f}\n")
#         observer.record_loss(epoch, loss_train, test_loss)
        
#         if observer.excute(epoch, model):       
#             print("Early stopping")
#             break
#     observer.finish()

def train_gcn(observer, patient_info, data, idx_train, idx_val, epochs, model, device, optimizer, criterion):
    print("start training")

    model = model.to(device) 
    
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)

    labels = torch.from_numpy(patient_info['label']).long().to(device) 
    
    # print(x[idx_val], labels[idx_val])
    # exit()

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        observer.reset()
        model.train()
        optimizer.zero_grad()
        output = model(x, edge_index, edge_attr)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss_train = criterion(output[idx_train], labels[idx_train])
        
        # loss_train.backward(retain_graph=True)
        loss_train.backward()
        optimizer.step()
        loss_train = loss_train.item()

        with torch.no_grad():
            model.eval()
            val_output = model(x, edge_index, edge_attr)
            loss_val = criterion(val_output[idx_val], labels[idx_val])

            outputs = val_output[idx_val]

            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(probabilities, dim=1)
            confidence_scores = probabilities[range(len(predictions)), 1]
            observer.update(predictions, labels[idx_val], confidence_scores)
            test_loss = loss_val.item() 
        observer.log(f"Train Loss: {loss_train:.4f}\n")
        observer.log(f"Test Loss: {test_loss:.4f}\n")
        observer.record_loss(epoch, loss_train, test_loss)
        
        if observer.excute(epoch, model):       
            print("Early stopping")
            break
    observer.finish()

"""
run single modal
"""
def run_single(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    observer.log("start training\n")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        running_loss = test_loss = 0.0
        for i, (info1, label) in enumerate(train_bar):
            optimizer.zero_grad()
            info1, label = info1.to(device), label.to(device)
            outputs = model(info1)
            loss = criterion(outputs, label)
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name, param.grad_fn)
            optimizer.step()
            running_loss += loss.item() * info1.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        observer.log(f"Loss: {train_loss:.4f}\n")

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (info1, label) in enumerate(test_bar):
                info1, label = info1.to(device), label.to(device)
                outputs = model(info1)
                loss = criterion(outputs, label)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(probabilities, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]
                observer.update(predictions, label, confidence_scores)
                test_loss += loss.item() * info1.size(0)
            test_loss = test_loss / len(train_loader.dataset)
            observer.log(f"Test Loss: {test_loss:.4f}\n")

        observer.record_loss(epoch, train_loss, test_loss)
        if observer.excute(epoch, model):
            print("Early stopping")
            break
    observer.finish()