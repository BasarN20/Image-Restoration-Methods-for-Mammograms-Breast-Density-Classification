import numpy as np 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import *
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import layers

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    
# Dice + Cross-Entropy loss fonksiyonunu tanımlayalım:
def dice_ce_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice + ce
"""
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_true = K.cast(y_true, tf.float32)
    loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred)
    loss -= (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
    return K.mean(loss)
"""
def focal_loss(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    y_pred = K.clip(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))
    loss -= (1 - y_true) * ((1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred))
    return K.mean(loss)
    
def combined_loss(y_true, y_pred):
    #return 0.5 * dice_loss(y_true, y_pred) + 0.5 * focal_loss(y_true, y_pred)
    return 0.5 * tversky_loss(y_true, y_pred) + 0.5 * focal_loss(y_true, y_pred)
    
def tversky(y_true, y_pred):
    smooth = 1e-15
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
"""
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
"""
def tversky_loss(y_true, y_pred):
    alpha = 0.3
    beta  =  0.7
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    TP = K.sum(y_true_f * y_pred_f)
    FP = K.sum((1 - y_true_f) * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))

    return 1 - ((TP + smooth) / (TP + alpha * FP + beta * FN + smooth))

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
"""
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
"""
def resize_y_true(y_true, y_pred):
    y_true = tf.image.resize(y_true, tf.shape(y_pred)[1:3])  
    return y_true

def updated_tversky_loss(y_true, y_pred):
    y_true = resize_y_true(y_true, y_pred)
    return 1 - tversky(y_true, y_pred)

def updated_dice_loss(y_true, y_pred):
    y_true = resize_y_true(y_true, y_pred)
    return 1 - dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
"""    
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss
"""
def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
"""
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)
"""
def iou(y_true, y_pred):
    smooth = 1e-6 
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    
    return (intersection + smooth) / (union + smooth)

def iou_loss(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1 - (intersection + smooth) / (union + smooth)

# Dice Loss 
def combined_iou_dice_loss(y_true, y_pred):
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# Precision 
def precision(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    true_positives = K.sum(y_true * y_pred)
    predicted_positives = K.sum(y_pred)
    
    return (true_positives + 1e-6) / (predicted_positives + 1e-6)

# Recall 
def recall(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    true_positives = K.sum(y_true * y_pred)
    actual_positives = K.sum(y_true)
    
    return (true_positives + 1e-6) / (actual_positives + 1e-6)

# F1-Score 
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-6)
    
def conv_block(input, num_filters, dropout_rate):
    x = Conv2D(num_filters // 2, 3, padding="same")(input)  
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters // 2, 3, padding="same")(x)  
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    return x
    
# Vision Transformer (ViT) Encoder
class ViTEncoder(layers.Layer):
    def __init__(self, patch_size=16, embed_dim=64, num_heads=4, num_layers=6, **kwargs):
        super(ViTEncoder, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.projection = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="same")
        self.transformer_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
            for _ in range(num_layers)
        ]
        self.norm = layers.LayerNormalization()

    def build(self, input_shape):
        
        self.cls_token = self.add_weight(
            "cls_token", shape=(1, 1, self.embed_dim), initializer="zeros", trainable=True
        )
        self.pos_embed = self.add_weight(
            "pos_embed", shape=(1, (480 // self.patch_size) * (240 // self.patch_size) + 1, self.embed_dim),
            initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = self.projection(x)  
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])  # [B, Patch, Embed]
        
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embed_dim])  # CLS token 
        x = tf.concat([cls_tokens, x], axis=1)  # [B, Patch + 1, Embed]
        x += self.pos_embed  

        for layer in self.transformer_layers:
            x = layer(x, x) + x
            x = self.norm(x)
        
        return x[:, 1:, :]  

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers
        })
        return config  


def decoder(x, output_channels):
    x = layers.Reshape((30, 15, -1))(x)  # Reshape into feature map (adjust as needed)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(output_channels, 3, strides=2, padding="same", activation="softmax")(x)
    return x

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Reshape, Activation
from tensorflow.keras.layers import Conv2D, Add, Activation, Multiply


def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, (1, 1), strides=1, padding='same')(x)
    phi_g = Conv2D(inter_channels, (1, 1), strides=1, padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    relu_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), strides=1, padding='same')(relu_xg)
    psi = Activation('sigmoid')(psi)
    attn_coeff = Multiply()([x, psi])
    return attn_coeff


def build_improved_vit_segmentation(input_shape=(480, 240, 3), num_classes=3):
    inputs = Input(shape=input_shape)
    
    # (skip connection )
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    pool_conv1 = MaxPooling2D(pool_size=(16, 16))(conv1)  # (480,240) -> (30,15)
    
    # ViT Encoder: 
    vit_features = ViTEncoder()(inputs)  #  (None, 450, embed_dim)
    vit_reshaped = Reshape((30, 15, -1))(vit_features)  # (None, 30,15, embed_dim)
    
    # Attention gate 
    attn_conv1 = attention_gate(pool_conv1, vit_reshaped, inter_channels=16)
    
    # ViT 
    combined = Concatenate()([vit_reshaped, attn_conv1])  # (None, 30,15, embed_dim+32)
    
    # Decoder: 
    x = Conv2DTranspose(128, (3, 3), strides=2, padding="same", activation="relu")(combined)   # (60,30,128)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)            # (120,60,64)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)            # (240,120,32)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding="same", activation="relu")(x)            # (480,240,16)
    
   
    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(x)                           # (480,240,3)
    
    return Model(inputs, outputs, name="Improved_ViT_Segmentation_3Classes")


model = build_improved_vit_segmentation()
model.summary()

