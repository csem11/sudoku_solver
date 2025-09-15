
import keras
from keras import layers


class DigitClassifier(keras.Model):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_param = input_shape
        self.num_classes = num_classes
        
        # Define layers
        self.conv1 = layers.Conv2D(64, (3, 3), activation="swish")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(0.5)
        
        # self.conv2 = layers.Conv2D(64, (3, 3), activation="swish")
        # self.bn2 = layers.BatchNormalization()
        # self.dense1 = layers.Dense(128, activation="swish")
        # self.dropout2 = layers.Dropout(0.4)
        
        self.conv3 = layers.Conv2D(32, (2, 2), activation="swish")
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.4)
        
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense2 = layers.Dense(128, activation="swish")
        self.dropout4 = layers.Dropout(0.5)
        self.output_layer = layers.Dense(num_classes, activation="softmax")
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        # x = self.conv2(x)
        # x = self.bn2(x, training=training)
        # x = self.dense1(x)
        # x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        
        x = self.global_avg_pool(x)
        x = self.dense2(x)
        x = self.dropout4(x, training=training)
        x = self.output_layer(x)
        
        return x
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_param,
            'num_classes': self.num_classes
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        # Extract our custom parameters
        input_shape = config.pop('input_shape', (28, 28, 1))
        num_classes = config.pop('num_classes', 10)
        
        # Create instance
        instance = cls(input_shape=input_shape, num_classes=num_classes)
        
        # Build the model with the input shape
        sample_input = keras.Input(shape=input_shape)
        _ = instance(sample_input)
        
        return instance