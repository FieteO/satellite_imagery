from tensorflow.keras.models import load_model
import tensorflow as tf
from helpers import (jaccard_coef,
                     jaccard_coef_int)



#set custom dependencies
dependencies = {
    'jaccard_coef' : jaccard_coef, 'jaccard_coef_int' : jaccard_coef_int
}

# open exported model in h5 format
model = load_model('unet_jk_score_0.443.h5', custom_objects=dependencies)

# save model as pb
tf.saved_model.save(model, '../unet/1')

#model.summary()

