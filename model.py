from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

def build_pretrained_model(
    base_model,
    hidden_dim=1024, 
    n_classes=6):

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(hidden_dim, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    preds = Dense(n_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=preds)
    
    return model