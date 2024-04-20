import torch
from torch import nn

def conv_block(in_ch, out_ch, stride):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
    )


def depthwise_conv_block(in_ch, out_ch, stride):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(num_features=in_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
    )


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        self.core_network = nn.Sequential(
            conv_block(3, 32, 2),
            depthwise_conv_block(32,  64, 1),
            depthwise_conv_block(64, 128, 2),
            depthwise_conv_block(128, 128, 1),
            depthwise_conv_block(128, 256, 2),
            depthwise_conv_block(256, 256, 1),
            depthwise_conv_block(256, 512, 2),
            depthwise_conv_block(512, 512, 1),
            depthwise_conv_block(512, 512, 1),
            depthwise_conv_block(512, 512, 1),
            depthwise_conv_block(512, 512, 1),
            depthwise_conv_block(512, 512, 1),
            depthwise_conv_block(512, 1024, 2),
            depthwise_conv_block(1024, 1024, 1),
            nn.AvgPool2d(7)
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 40)
        )

    def forward(self, inp):
        inp = self.core_network(inp)
        inp = torch.flatten(inp, 1)
        inp = self.classifier_head(inp)
        return inp


    def fit(self, train_data, optimizer, loss_fn, epochs):
        self.train(mode=True)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            for i, batch in enumerate(train_data):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                result = outputs > 0.5

                acc = (result == labels).sum().item()
                epoch_accuracy += acc
                epoch_loss += loss.item()
                # loss per batch
                batch_acc = (acc / (len(images) * labels.shape[1])) * 100.

                # Measure model performance for every batch
                #calc_evaluation_metrics(labels.cpu().detach().numpy(), result.cpu().detach().numpy(), epoch, i)

            epoch_accuracy /= len(train_data)
            epoch_loss /= len(train_data)
            #print('Epoch {} is completed. Loss: {}, Accuracy: {}'.format(epoch + 1, epoch_loss, epoch_accuracy))

