import theano
print theano.config.device
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Merge, Dropout, Input, Masking, merge, LSTM, Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, Flatten
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import sys

input_shapes = [(6,20),(6,20),(25,20),(1,20),(100,20),(100,20),(100,20)]
inputs = []
for shape in input_shapes:
    inputs.append( Input(shape = shape) )

merged = merge(inputs, mode='concat', concat_axis=1)
masked = Masking()(merged)

label = sys.argv[1]

if label == 'onelstm':
    lstm1 = LSTM(10)(masked)
    last_layer = lstm1
elif label == 'twolstm':
    lstm1 = LSTM(10,return_sequences=True)(masked)
    lstm2 = LSTM(10)(lstm1)
    last_layer = lstm2
elif label == 'cnn':
    cnn = Convolution1D(10, 10)(merged)
    maxp = MaxPooling1D(5)(cnn)
    last_layer = Dense(20)(Flatten()(maxp))
else:
    print label,"not recognized"
    
output = Dense(3, activation='softmax')(last_layer)

model = Model( input =  inputs, output = output)
model.compile( optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

model.summary()

open('%s_arch.json'%label,'w').write(model.to_json())
model.save('%s.h5'%label)
