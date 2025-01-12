import torch
import torch.nn as nn
import torch
import torch.nn as nn

class LSTMModel(nn.Module): #MAYBE CONVERT TO CONV3d
    def __init__(self, batchSize = 1, hiddenSize = 64, numLayers = 1, output_size = 256, num_frames = 60):
        super(LSTMModel, self).__init__()
        self.drop0 = nn.Dropout(0.1)
        self.lin_0_0 = nn.Linear(21*3, 21*3)
        self.lin_0_1 = nn.Linear(21*3, 64)
        self.lstm = nn.LSTM(64+252, hiddenSize, numLayers)
        self.lin_1_0 = nn.Linear(hiddenSize, 32)
        self.lin_1_1 = nn.Linear(32,32)
        self.lin_1_2 = nn.Linear(32, output_size)
        self.norm0 = nn.BatchNorm1d(60, affine = True)
        self.norm1 = nn.BatchNorm1d(60, affine = True)
        self.norm2 = nn.BatchNorm1d(60, affine = True)
        self.norm3 = nn.BatchNorm1d(60, affine = True)
        
        self.conv0 = nn.Conv2d(1, 16, 5, 4)
        self.conv1 = nn.Conv2d(16, 16, 5,4)
        self.conv2 = nn.Conv2d(16,4, 2, 2)
        self.drop1 = nn.Dropout(0.5)   
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.6)

        self.hidden = torch.zeros(1, hiddenSize)
        self.cell = torch.zeros(1, hiddenSize)
        self.hiddenSize = hiddenSize
        self.num_frames = num_frames
        #Need h and c

    def forward(self, input, video):
        self.hidden = torch.zeros(1, self.hiddenSize).to('cuda')
        self.cell = torch.zeros(1, self.hiddenSize).to('cuda')
        output = None
        for i in range(self.num_frames):
            input_frame = nn.functional.relu(self.lin_0_0(input[:,i,:]))
            input_frame = self.drop1(input_frame)
            input_frame = nn.functional.relu(self.lin_0_1(input_frame))
            input_image = nn.functional.relu(self.conv2(nn.functional.relu(self.conv1(nn.functional.relu(self.conv0(video[:,i,:,:].unsqueeze(1)))))))
            input_image = torch.flatten(input_image,start_dim=1)
            input_frame = torch.concat((input_frame, input_image), dim=1)
            output, (self.hidden, self.cell) = self.lstm(input_frame, (self.hidden, self.cell))
        output = self.drop2(output)
        output = nn.functional.relu(self.lin_1_0(output))
        output = nn.functional.relu(self.lin_1_1(output))
        return self.lin_1_2(output)

    def predict(self, input, prevH, prevC):
        output, (hn, cn) = self.lstm(input, (prevH, prevC))
        return self.lin(output), (hn, cn)
    


class LSTMModel_No_Image(nn.Module):
    def __init__(self, batchSize = 1, hiddenSize = 32, numLayers = 1, output_size = 256, num_frames = 60):
        super(LSTMModel_No_Image, self).__init__()
        self.drop0 = nn.Dropout(0.1)
        self.lin_0_0 = nn.Linear(21*3, 21*3)
        self.lin_0_1 = nn.Linear(21*3, 64)
        self.lstm = nn.LSTM(64, hiddenSize, numLayers)
        self.lin_1_0 = nn.Linear(hiddenSize, 32)
        self.lin_1_1 = nn.Linear(32,32)
        self.lin_1_2 = nn.Linear(32, output_size)
        self.layer_norm = nn.LayerNorm(64)
        self.norm0 = nn.BatchNorm1d(hiddenSize, affine = True)
        self.drop1 = nn.Dropout(0.5)   
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.6)

        self.hidden = torch.zeros(1, hiddenSize)
        self.cell = torch.zeros(1, hiddenSize)
        self.hiddenSize = hiddenSize
        self.num_frames = num_frames
        #Need h and c

    def forward(self, input):
        self.hidden = torch.zeros(1, self.hiddenSize).to('cuda')
        self.cell = torch.zeros(1, self.hiddenSize).to('cuda')
        output = None
        for i in range(self.num_frames):
            input_frame = nn.functional.relu(self.lin_0_0(input[:,i,:]))
            input_frame = nn.functional.relu(self.lin_0_1(input_frame))
            input_frame = self.layer_norm(input_frame)
            output, (self.hidden, self.cell) = self.lstm(input_frame, (self.hidden, self.cell))
        output = self.norm0(output)
        print(output.shape)
        output = nn.functional.relu(self.lin_1_0(output))
        output = nn.functional.relu(self.lin_1_1(output))
        return self.lin_1_2(output)

    def predict(self, input, prevH, prevC):
        output, (hn, cn) = self.lstm(input, (prevH, prevC))
        return self.lin(output), (hn, cn)


class LSTMModel_No_Image_Critique(nn.Module):
    def __init__(self, batchSize = 1, hiddenSize = 32, numLayers = 1, output_size = 256, num_frames = 60):
        super(LSTMModel_No_Image_Critique, self).__init__()
        self.drop0 = nn.Dropout(0.1)
        self.lin_0_0 = nn.Linear(21*3, 21*3)
        self.lin_0_1 = nn.Linear(21*3, 64)
        self.lin_0_2 = nn.Linear(64, 64)
        self.lstm = nn.LSTM(64, hiddenSize, numLayers, batch_first= True)
        self.lin_1_0 = nn.Linear(hiddenSize, 32)
        self.lin_1_1 = nn.Linear(32,32)
        self.lin_1_2 = nn.Linear(32, output_size)
        self.layer_norm = nn.LayerNorm(64)
        self.norm0 = nn.BatchNorm1d(hiddenSize, affine = True)
        self.drop1 = nn.Dropout(0.25)   
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.6)

        self.hidden = torch.zeros(1, hiddenSize)
        self.cell = torch.zeros(1, hiddenSize)
        self.hiddenSize = hiddenSize
        self.num_frames = num_frames
        #Need h and c

    def forward(self, input):
        batch_size, num_frames, input_size = input.shape
        #self.hidden = torch.zeros(batch_size, self.hiddenSize).to('cuda')
        #self.cell = torch.zeros(batch_size, self.hiddenSize).to('cuda')
        output = None
        input_reshaped = input.reshape(batch_size*num_frames,-1)
        input_frame = nn.functional.relu(self.lin_0_0(input_reshaped))
        input_frame = nn.functional.relu(self.lin_0_1(input_frame))
        input_frame = nn.functional.relu(self.lin_0_2(input_frame))
        input_frame= self.layer_norm(input_frame)
        processed_frames = input_frame.reshape(batch_size, num_frames, -1)
        output, (self.hidden, self.cell) = self.lstm(processed_frames)#, (self.hidden, self.cell))
        output = output[:,-1,:]
        output = self.norm0(output)
        output = nn.functional.relu(self.lin_1_0(output))
        output = nn.functional.relu(self.lin_1_1(output))
        return self.lin_1_2(output)

    def predict(self, input, prevH, prevC):
        output, (hn, cn) = self.lstm(input, (prevH, prevC))
        return self.lin(output), (hn, cn)


class LSTMModel_AutoEncode(nn.Module):
    def __init__(self, hiddenSize=32, numLayers=1, num_frames=60, z_dim=2):
        super(LSTMModel_AutoEncode, self).__init__()
        
        # Linear layers to process input z
        self.lin_0_0 = nn.Linear(z_dim, 21*3)  # Input z_dim -> 21*3
        self.lin_0_1 = nn.Linear(21*3, 64)
        self.lin_0_2 = nn.Linear(64, 64)
        
        # LSTM to process the features and generate a sequence of frames
        self.lstm = nn.LSTM(64, hiddenSize, numLayers, batch_first=True)
        
        # Fully connected layers after LSTM to map the hidden states to the output size (63)
        self.lin_1_0 = nn.Linear(hiddenSize, 32)
        self.lin_1_1 = nn.Linear(32, 32)
        self.lin_1_2 = nn.Linear(32, 63)  # Final output size (63)
        
        # Normalization layers
        self.layer_norm = nn.LayerNorm(64)
        
        # Dropout layers for regularization
        self.drop1 = nn.Dropout(0.25)   
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.6)

        # Store dimensions
        self.hiddenSize = hiddenSize
        self.num_frames = num_frames
        self.z_dim = z_dim

    def forward(self, input):
        batch_size = input.shape[0]

        # Process the latent vector z (size: batch_size, z_dim) through the linear layers
        z = nn.functional.relu(self.lin_0_0(input))  # First linear layer
        z = nn.functional.relu(self.lin_0_1(z))     # Second linear layer
        z = nn.functional.relu(self.lin_0_2(z))     # Third linear layer
        z = self.layer_norm(z)                      # Layer normalization

        # Reshape z to match the LSTM input (batch_size, num_frames, feature_size)
        z = z.unsqueeze(1).repeat(1, self.num_frames, 1)  # Repeat across 60 frames

        # Pass the processed z through the LSTM to generate the sequence of frames
        lstm_output, (hidden, cell) = self.lstm(z)

        # Pass through fully connected layers to map LSTM outputs to the desired output size (63)
        output = nn.functional.relu(self.lin_1_0(lstm_output))  # First linear layer after LSTM
        output = nn.functional.relu(self.lin_1_1(output))      # Second linear layer after LSTM
        output = self.lin_1_2(output)                          # Final output layer (size: 63)

        return output

    def predict(self, input, prevH, prevC):
        # This method is for step-by-step prediction (autoregressive), if needed
        output, (hn, cn) = self.lstm(input, (prevH, prevC))
        return self.lin_1_2(output), (hn, cn)


class NNModel(nn.Module):
    def __init__(self, input_size = 63, output_size = 3):
        super(NNModel, self).__init__()
        self.drop0 = nn.Dropout(0.1)
        self.lin0 = nn.Linear(input_size, 32)
        self.lin = nn.Linear(32, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, output_size)
        self.drop1 = nn.Dropout(0.1)   
        self.drop2 = nn.Dropout(0.1)

    def forward(self, input):
        output = nn.functional.relu(self.lin0(input))
        output = self.drop1(output)
        output= self.lin(output)
        output= self.drop2(output)
        #return self.lin2(nn.functional.relu(output))
        return self.lin3(nn.functional.relu(self.lin2(nn.functional.relu(output))))
    





class OLD_LSTMModel(nn.Module):
    def __init__(self, batchSize = 1, hiddenSize = 150, numLayers = 2, output_size = 5):
        super(OLD_LSTMModel, self).__init__()
        self.drop0 = nn.Dropout(0.1)
        self.lin0 = nn.Linear(21*3, 21*3)
        self.lin01 = nn.Linear(21*3, 256)
        self.lstm = nn.LSTM(256, hiddenSize, numLayers, dropout = 0.5, batch_first=True)
        self.lin = nn.Linear(hiddenSize, 256)
        self.lin2 = nn.Linear(256, 126)
        self.lin3 = nn.Linear(126, 126)
        self.lin4 = nn.Linear(126, output_size)
        self.norm0 = nn.BatchNorm1d(60, affine = True)
        self.norm1 = nn.BatchNorm1d(60, affine = True)
        self.norm2 = nn.BatchNorm1d(60, affine = True)
        self.norm3 = nn.BatchNorm1d(60, affine = True)
        
        self.drop1 = nn.Dropout(0.5)   
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.6)

    def forward(self, input):
        output = nn.functional.relu(self.lin0(input))
        layer0 = self.lin01(output)
        output = nn.functional.relu(layer0)
        output = self.norm0(output)
        output = self.drop0(output)
        output, _ = self.lstm(output)
        output = self.norm3(output)
        output = self.drop1(output)
        output= self.lin(output) + layer0
        output = self.drop2(output)
        output = self.norm1(output)
        output = self.lin2(nn.functional.relu(output))
        output = self.norm2(output)
        #return self.lin2(nn.functional.relu(output))
        return self.lin4(nn.functional.relu(output))

    def predict(self, input, prevH, prevC):
        output, (hn, cn) = self.lstm(input, (prevH, prevC))
        return self.lin(output), (hn, cn)
    

class NNModel(nn.Module):
    def __init__(self, input_size, output_size = 3):
        super(NNModel, self).__init__()
        self.drop0 = nn.Dropout(0.1)
        self.lin0 = nn.Linear(input_size, 64)
        self.lin1 = nn.Linear(64, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 32)
        self.lin4 = nn.Linear(32, output_size)
        self.drop1 = nn.Dropout(0.1)   
        self.drop2 = nn.Dropout(0.2)


    def forward(self, input):
        output = nn.functional.relu(self.lin0(input))
        output = self.drop1(output)
        output= self.lin1(output)
        output = self.bn1(output)
        #return self.lin2(nn.functional.relu(output))
        return self.lin4(nn.functional.relu(self.drop2(self.lin3(nn.functional.relu(self.lin2(nn.functional.relu(output)))))))
    
