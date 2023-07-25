from unet_model import *
from Tracker import *
from plot_parts import plot_images_batch
from torch.utils.data import DataLoader





SNR = 20
file_name = f'Emission_DNN/dnn_state_SNR' + str(SNR)


emis_DNN = UNet(n=64,
                bilinear=True)
emis_DNN.load_state_dict(torch.load(file_name))


data = torch.load('track_data_'+str(SNR)+'SNR')
data.Nd = 64

T = 1


BBox = [60,10,200,54]

batch_size = 1
loader = DataLoader(data,batch_size=batch_size,shuffle=True)

acc_vit_final = 0
acc_SFD_final = 0

big = 0
transdist = TransDist(sigma_r=30,sigma_v=20,T=0)


emis_DNN.train()
viterbi_avg = 0
sfd_avg = 0

viterbi_avg_soft = 0
sfd_avg_soft = 0
with torch.no_grad():
    for i, (input, label) in enumerate(loader):
        viterbi = Tracker(transdist)
        label = label


        input = input.transpose(0,1)
        output = emis_DNN(input,bbox=BBox,restore=True)
        output = output.squeeze(1)

        #estim_hough_1 = viterbi.HoughTacker(input.squeeze(1))
        #print(estim_hough_1)
        #hough_1 = viterbi.Find_Accuracy(estim_hough_1,label,"Hough Transform (observation)")

        #estim_hough_2 = viterbi.HoughTacker(output)
        #print(estim_hough_2)
        #hough_2 = viterbi.Find_Accuracy(estim_hough_1,label,"Hough Transform (LL)")

        estim_SFD = viterbi.SingleFrameDetection(output)
        #print(estim_SFD)
        acc_SFD, acc_sfd_soft = viterbi.Find_Accuracy(estim_SFD,label,"SFD")

        estim_vit = viterbi.backward_viterbi(output,num_tracks=30)
        #print(estim_vit)
        acc_vit, acc_vit_soft = viterbi.Find_Accuracy(estim_vit,label,"Viterbi")

        viterbi_avg += acc_vit
        viterbi_avg_soft += acc_vit_soft
        sfd_avg += acc_SFD
        sfd_avg_soft += acc_sfd_soft
        plot_images_batch(input,label,output,estimated_viterbi=estim_vit,estimated_SFD=estim_SFD,save_path='cool_dudes.gif')


print(f'Average SFD accuracy =  {sfd_avg*100/10: .2f} %')
print(f'Average SFD soft accuracy =  {sfd_avg_soft*100/10: .2f} %')

print(f'Average Viterbi accuracy = {viterbi_avg*100/10: .2f} %')
print(f'Average Viterbi soft accuracy = {viterbi_avg_soft*100/10: .2f} %')