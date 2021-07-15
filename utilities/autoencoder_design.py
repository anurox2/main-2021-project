from itertools import count
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import losses
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import logging
import inspect, traceback
from .logging import custom_file_handler

file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

design_structures = {
    '24_8_3': '24_8_3',
    '24_12_6_3': '24_12_6_3',
    '24_18_12_8_6_3': '24_18_12_8_6_3',
}

def design(design_key, X, loss_function, activation='linear'):
    design_structure = design_structures[design_key]
    design_list = design_structure.split('_')

    number_of_inputs = X.shape[1]
    input_encoder = Input(shape=(number_of_inputs,))
    logger.info('Input encoder created')
    
    # encoder layer creation
    first_layer = True
    list_of_encoders = []
    for level in design_list[:-1]:
        encoder = None
        if(first_layer):
            first_layer = False
            encoder = Dense(int(level))(input_encoder)
            logger.info('First layer of encoder created with {0} nodes'.format(level))
        else:
            encoder = Dense(int(level))(list_of_encoders[-1])
        
        encoder = BatchNormalization()(encoder)
        encoder = LeakyReLU()(encoder)
        logger.info('Encoder layer created with {0} nodes'.format(level))
        list_of_encoders.append(encoder)
        logger.info('Number of encoder layers: {0}'.format(len(list_of_encoders)))
        del encoder

    bottleneck = Dense(design_list[-1])(list_of_encoders[-1])
    design_list.reverse()

    first_layer = True
    list_of_decoders = []
    for level in design_list[1:]:
        decoder = None
        if(first_layer):
            first_layer = False
            decoder = Dense(int(level))(bottleneck)
            logger.info('First layer of decoder created with {0} nodes'.format(level))
        else:
            decoder = Dense(int(level))(list_of_decoders[-1])
        decoder = BatchNormalization()(decoder)
        decoder = LeakyReLU()(decoder)
        logger.info('Decoder layer created with {0} nodes'.format(level))
        list_of_decoders.append(decoder)
        logger.info('Number of decoder layers: {0}'.format(len(list_of_encoders)))
        del decoder
    
    output = Dense(design_list[-1], activation=activation)(list_of_decoders.pop())
    logger.info('Output layer created')

    # Defining the autoencoder model
    model = Model(inputs=input_encoder, outputs=output)
    model.compile(Adam(0.001), loss=loss_function)
    plot_model(model, "autoencoder.png", show_shapes=True)
    logger.info('Autoencoder model created and structure plotted')

    # define an encoder model (without the decoder)
    encoder_no_decoder = Model(inputs=input_encoder, outputs=bottleneck)
    plot_model(encoder_no_decoder, 'encoder.png', show_shapes=True)
    logger.info('Encoder "only" model created and structure plotted')

    logger.info('Returning models to main')
    return model, encoder_no_decoder



# def design_autoencoder(X, neurons, loss_function, activation='linear'):
# # Encoder design
#     number_of_inputs = X.shape[1]
#     input_encoder = Input(shape=(number_of_inputs,))

# ## Encoder level 1
#     encoder_1 = Dense(number_of_inputs*2)(input_encoder)
#     encoder_1 = BatchNormalization()(encoder_1)
#     encoder_1 = LeakyReLU()(encoder_1)

# ## Encoder level 2
#     encoder_2 = Dense(number_of_inputs)(encoder_1)
#     encoder_2 = BatchNormalization()(encoder_2)
#     encoder_2 = LeakyReLU()(encoder_2)

# #
#     divisor = number_of_inputs/neurons

# # Bottleneck design
#     bottleneck_inputs = number_of_inputs/divisor
#     bottleneck = Dense(bottleneck_inputs)(encoder_2)

# # Decoder design
# ## Decoder Level 1
#     decoder_1 = Dense(number_of_inputs)(bottleneck)
#     decoder_1 = BatchNormalization()(decoder_1)
#     decoder_1 = LeakyReLU()(decoder_1)

# ## Decoder Level 2
#     decoder_2 = Dense(number_of_inputs*2)(decoder_1)
#     decoder_2 = BatchNormalization()(decoder_2)
#     decoder_2 = LeakyReLU()(decoder_2)

# #
#     output = Dense(number_of_inputs, activation=activation)(decoder_2)

#     # Defining the autoencoder model
#     model = Model(inputs=input_encoder, outputs=output)
#     model.compile(Adam(0.001), loss=loss_function)
#     plot_model(model, "autoencoder.png", show_shapes=True)

#     # define an encoder model (without the decoder)
#     encoder_no_decoder = Model(inputs=input_encoder, outputs=bottleneck)
#     plot_model(encoder_no_decoder, 'encoder.png', show_shapes=True)

#     return model, encoder_no_decoder