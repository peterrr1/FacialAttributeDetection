from torch import nn


def conv_block(in_ch, out_ch):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
    )


def depthwise_conv_block(in_ch, out_ch):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=(3, 3), padding=1, stride=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(num_features=in_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, stride=1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
    )


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        self.core_network = nn.Sequential(
            conv_block(3, 32),
            depthwise_conv_block(32, 64),
            depthwise_conv_block(64, 128),
            depthwise_conv_block(128, 128),
            depthwise_conv_block(128, 256),
            #depthwise_conv_block(256, 256),
            #depthwise_conv_block(256, 512),
            #depthwise_conv_block(512, 512),
            #depthwise_conv_block(512, 1024),
            #depthwise_conv_block(1024, 1024),
            nn.AvgPool2d(kernel_size=(7, 7))
        )
        self.fully_connected = nn.Linear(128, 40)
        self.classifier = nn.Softmax()

    def forward(self, inp):
        inp = self.core_network(inp)
        inp = inp.view(inp.size(0), -1)
        inp = self.fully_connected(inp)
        inp = self.classifier(inp)
        return inp

    def fit(self, train_data, optimizer, loss_fn, epochs):
        self.train(mode=True)
        running_loss = 0.0
        correct = 0

        for i, batch in enumerate(train_data):
            images, labels = batch
            optimizer.zero_grad()
            outputs = self.forward(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            result = outputs > 0.5
            correct += (result == labels).sum().item()

            print('Training set: [Epoch: %d, Data: %6d] Loss: %.3f' % (epochs + 1, i * 16, loss.item()))

        acc = correct / (len(train_data) * 40)
        running_loss /= len(train_data)
        print('\nTraining set: Epoch: %d, Accuracy: %.2f %%' % (epochs + 1, 100. * acc))
