from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

#Define the model 
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape= (150,150,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(
    optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy']
    )


# Define the data 
train = ImageDataGenerator(rescale = 1./255,shear_range = 0.2 , zoom_range = 0.2 , horizontal_flip = True)
test = ImageDataGenerator(rescale = 1./255)
train_img = train.flow_from_directory('C://PROJECTS//Face_Mask_Detection//data//train',target_size = (150,150),batch_size = 16 , class_mode = 'binary')
test_img = test.flow_from_directory('C://PROJECTS//Face_Mask_Detection//data//test',target_size = (150,150),batch_size = 16 , class_mode = 'binary')

# Train & Test the model
mask_model = model.fit(train_img,epochs=10,validation_data=test_img)

# Save the model
model.save('Mask.h5')