"""

Only Trainer can train and evaluate model

"""

import matplotlib.pyplot as plt

import torch
from torch import nn, optim


class Trainer:
    def __init__(self, model_manager, data_manager, cfg, writer, arti_path):
        self.model_manager = model_manager
        self.data_manager = data_manager
        self.device = cfg.device
        self.criterion_img = nn.MSELoss().to(self.device)
        self.criterion_cnn = nn.CrossEntropyLoss().to(self.device)
        self.max_iteration = cfg.max_iteration
        self.max_iteration_retrain = cfg.max_iteration_retrain
        self.use_template = cfg.use_template
        self.verbose = cfg.verbose
        self.writer = writer
        self.arti_path = arti_path

    def _one_step(self, data, use_template=False):
        x = data[0].to(self.device)
        y = data[1].to(self.device)
        self.optimizer.zero_grad()
        logits, tmp = self.model(x)
        loss = self.criterion_cnn(logits, y)
        if use_template:
            template = data[2].to(self.device)
            loss2 = self.criterion_img(tmp, template)
            loss = loss + self.model.actor.reg_coef*loss2
        loss.backward()
        self.optimizer.step()
        return loss, tmp[0], y[0]
    
    def _train_endtoend(self, train_loader, valid_loader):
        max_acc, epoch, ite = -1, 0, 0
        max_epoch = self.max_iteration / len(train_loader)
        self.optimizer = optim.Adam(self.model.parameters())
        while ite < self.max_iteration:
            for data in train_loader:
                loss, tmp, label = self._one_step(data, self.use_template)
                ite += 1
                self.model.actor.attention.ite = ite
                if ite == self.max_iteration:
                    break       

            epoch += 1
            if epoch % (max_epoch//5) == 0:
                self._save_img(tmp, label, epoch)

            temp_flag = self.model.actor.attention.temperature < 1.0
            max_acc = self.report_and_log(epoch, max_acc, train_loader, valid_loader, flag=temp_flag)

        return max_acc

    def _train(self, h_param, train_loader, valid_loader):
        self.model = self.model_manager.get_model(h_param)
        self.model.to(self.device)
        acc = self._train_endtoend(train_loader, valid_loader)
            
        return acc

    def train(self, cfg, X_train, y_train, X_valid, y_valid):
        train_loader, valid_loader = self.data_manager.make_loader(X_train,
                                                                   y_train,
                                                                   X_valid,
                                                                   y_valid,
                                                                   use_template=self.use_template)
        self._train(cfg, train_loader, valid_loader)

    def _retrain(self, train_loader, valid_loader):
        max_acc, epoch, ite = -1, 0, 0
        for i in self.model.actor.parameters():
            i.requires_grad = False

        while ite < self.max_iteration_retrain:
            self.model.actor.attention.eval()
            for data in train_loader:
                loss, _, _ = self._one_step(data)
                ite += 1
                if ite == self.max_iteration_retrain:
                    break

            epoch += 1
            max_acc = self.report_and_log(epoch, max_acc, train_loader, valid_loader, key="re_")

        return max_acc

    def retrain(self, X_train, y_train, X_valid, y_valid):
        train_loader, valid_loader = self.data_manager.make_loader(X_train,
                                                                   y_train,
                                                                   X_valid,
                                                                   y_valid,
                                                                   use_template=self.use_template)

        self.load_model()
        self._retrain(train_loader, valid_loader)

        return 1
        
    def test(self, X_train, y_train, X_test, y_test):
        train_loader, test_loader = self.data_manager.make_loader(X_train,
                                                                  y_train,
                                                                  X_test,
                                                                  y_test,
                                                                  use_template=False)

        self.load_model()
        train_acc, _ = self.calc_acc(train_loader)
        test_acc, _ = self.calc_acc(test_loader)

        return train_acc, test_acc

    def calc_acc(self, loader):
        correct, total, loss = 0, 0, 0
        self.model.eval()
        with torch.inference_mode():
            for data in loader:
                x, y = data[0], data[1]
                x = x.to(self.device)
                y = y.to(self.device)
                logits, _ = self.model(x)
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                loss += self.criterion_cnn(logits, y)*y.size(0)
        acc = 100 * float(correct/total)
        self.model.train()
        return acc, loss.item()/total

    def _save_img(self, data, label, epoch):
        fig = plt.figure()
        plt.imshow(data.detach().cpu().numpy(), cmap='gray')
        plt.title(f'{label.detach().cpu().numpy()}')
        plt.axis('off')
        self.writer.log_figure(fig, f'im{epoch}.png')
        plt.close(fig)

    def report_and_log(self, epoch, max_acc, train_loader, valid_loader, flag=True, key=""):
        t_acc, t_loss = self.calc_acc(train_loader)
        v_acc, v_loss = self.calc_acc(valid_loader)
        if flag and v_acc >= max_acc:
            max_acc = v_acc
            self.log_model()

        if self.verbose:
            monitor = {key+'train_loss':t_loss,
                       key+'train_acc':t_acc,
                       key+'validation_loss':v_loss,
                       key+'validation_acc':v_acc,
                       key+'temp': self.model.actor.attention.temperature}
            print(f'epoch:{epoch}', end=' ')
            for key, item in monitor.items():
                print(f'{key}:{item:.3f}', end=' ')
                self.writer.log_metric_step(key, item, step=epoch)
            print()
            
        return max_acc

    def log_model(self, name="best"):
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.arti_path + f'/{name}_checkpoint.pth')

    def load_model(self, name="best"):
        try:
            dic = torch.load(self.arti_path+f'/{name}_checkpoint.pth')
            self.optimizer.load_state_dict(dic['optimizer_state_dict'])
            self.model.load_state_dict(dic['model_state_dict'])
            self.model.to(self.device)
        except Exception as e:
            print(e)
            print("Last epoch model will be used.")
            
    def save(self, ):
        torch.save(self.model.actor.state_dict(), self.arti_path+'/actor.pth')
        torch.save(self.model.critic.state_dict(), self.arti_path+'/critic.pth')
        torch.save(self.model.state_dict(), self.arti_path+'/model.pth')