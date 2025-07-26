import tensorflow as tf

# 📌 Dice metric
def dice_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, tf.shape(y_pred)[-1])
    y_true = tf.reshape(y_true, [-1, tf.shape(y_pred)[-1]])
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (denominator + smooth)

# 📌 Dice loss
def dice_loss(y_true, y_pred):
    return 1.0 - dice_metric(y_true, y_pred)

# 📌 Dice loss as metric
def dice_loss_metric(y_true, y_pred):
    return dice_loss(y_true, y_pred)

# 📌 Cross entropy
def balanced_cross_entropy(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

# 📌 Cross entropy as metric
def cross_entropy_metric(y_true, y_pred):
    return balanced_cross_entropy(y_true, y_pred)

# 📌 Total loss = Dice + Cross Entropy
def total_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + balanced_cross_entropy(y_true, y_pred)

# 📌 Wrapper pour MeanIoU compatible Keras
class MeanIoUWrapper(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=8, name='mean_iou', **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)  # (batch, H, W)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

# ✅ Exporter cette classe pour import
mean_iou = MeanIoUWrapper(num_classes=8)