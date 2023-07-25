
from train_parts import *
from plot_parts import plot_loss_and_accuracy


batch_size = 100


# Getting ready for training DNN
emis_DNN = UNet(n=64,
                bilinear=True)
# load existing DNN
if not os.path.exists('Emission_DNN'):
    os.makedirs('Emission_DNN')

answer = input("To train a new DNN enter 'N', To continue training the last DNN enter 'C': ")
if answer.lower() == "c":
    emis_DNN.load_state_dict(torch.load(f"emission_DNN/dnn_state"))





train_data = torch.load('valid_data_10')
valid_data = torch.load('valid_data_10')
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)
valid_loader = DataLoader(valid_data,
                          batch_size=len(valid_data),
                          shuffle=True)

file_name = f'Emission_DNN/dnn_state'

train_loss,train_acc,valid_loss,valid_acc = train_model(emis_DNN,
                                                        opt = "adam",
                                                        train_loader=train_loader,
                                                        valid_loader=valid_loader,
                                                        learning_rate=0.001,
                                                        epochs=75,
                                                        weight_decay=1)
plot_loss_and_accuracy(train_loss,
                       train_acc,
                       valid_loss,
                       valid_acc,
                       filename='fun time')