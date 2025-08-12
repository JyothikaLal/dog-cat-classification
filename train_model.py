import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc

# GPU setup with memory limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limit GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Optionally set memory limit (uncomment next 2 lines if needed)
            # tf.config.experimental.set_memory_limit(gpu, 4096)  # 4GB limit
        print(f"GPU found: {gpus}")
    except RuntimeError as e:
        print(f"GPU error: {e}")
else:
    print("No GPU found, using CPU")

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

def create_data_generators(data_path, batch_size=16, img_size=(224, 224)):
    """Create memory-efficient data generators"""
    print("Creating data generators...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )
    
    # No augmentation for validation/test
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_input
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_path, 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def load_small_test_batch(data_path, num_samples=100):
    """Load a small batch for visualization"""
    images = []
    labels = []
    
    test_path = os.path.join(data_path, 'test')
    
    # Try both directory structures
    cats_path = os.path.join(test_path, 'cats')
    dogs_path = os.path.join(test_path, 'dogs')
    
    if not os.path.exists(cats_path):
        cats_path = os.path.join(test_path, 'Cat')
    if not os.path.exists(dogs_path):
        dogs_path = os.path.join(test_path, 'Dog')
    
    count = 0
    
    # Load cats
    if os.path.exists(cats_path) and count < num_samples:
        cat_files = [f for f in os.listdir(cats_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for filename in cat_files[:num_samples//2]:
            try:
                img_path = os.path.join(cats_path, filename)
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(1)  # cat=1
                count += 1
                if count >= num_samples//2:
                    break
            except:
                continue
    
    # Load dogs
    if os.path.exists(dogs_path) and count < num_samples:
        dog_files = [f for f in os.listdir(dogs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for filename in dog_files[:num_samples//2]:
            try:
                img_path = os.path.join(dogs_path, filename)
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(0)  # dog=0
                count += 1
                if count >= num_samples:
                    break
            except:
                continue
    
    if len(images) == 0:
        return None, None
    
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)
    
    return images, labels

def create_model():
    """Create lightweight ResNet50 model"""
    print("Creating ResNet50 model...")
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    # Simpler architecture to save memory
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)  # Reduced from 128
    x = Dropout(0.1)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model created!")
    return model

def plot_training_history(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory

def evaluate_model_generator(model, test_generator):
    """Evaluate model using generator"""
    print("Evaluating model...")
    
    # Reset generator
    test_generator.reset()
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Dog', 'Cat']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Dog', 'Cat'], yticklabels=['Dog', 'Cat'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory
    
    return accuracy

def plot_sample_predictions(model, X_test, y_test, num_samples=8):
    """Plot sample predictions with fewer samples"""
    class_names = ['Dog', 'Cat']
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = class_names[y_test[idx]]
        
        pred_proba = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_label = class_names[np.argmax(pred_proba)]
        confidence = np.max(pred_proba) * 100
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', fontsize=9)
        axes[i].axis('off')
        
        if true_label == pred_label:
            axes[i].patch.set_edgecolor('green')
        else:
            axes[i].patch.set_edgecolor('red')
        axes[i].patch.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory

def main():
    print("🐕🐱 Memory-Optimized ResNet Training 🐱🐕")
    
    # Check data
    data_path = 'data_split'
    if not os.path.exists(data_path):
        print("Data not found! Run split_data.py first")
        return
    
    # Use smaller batch size to save memory
    batch_size = 16  # Reduced from 32
    
    # Create data generators (memory efficient)
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators(data_path, batch_size)
    
    print(f"\nDataset:")
    print(f"Train: {train_gen.samples} images")
    print(f"Val: {val_gen.samples} images") 
    print(f"Test: {test_gen.samples} images")
    
    # Create model
    model = create_model()
    
    # Train with generators
    print("\nTraining...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    ]
    
    steps_per_epoch = train_gen.samples // batch_size
    validation_steps = val_gen.samples // batch_size
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=20,  # Reduced epochs
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating...")
    accuracy = evaluate_model_generator(model, test_gen)
    
    # Load small batch for visualization
    print("Loading small test batch for visualization...")
    X_test_small, y_test_small = load_small_test_batch(data_path, 50)
    
    if X_test_small is not None:
        plot_sample_predictions(model, X_test_small, y_test_small, 8)
        
        # Free memory
        del X_test_small, y_test_small
        gc.collect()
    
    # Save model
    model.save('dog_cat_model.h5')
    print("Model saved as dog_cat_model.h5")
    
    print(f"\n🎉 Training completed! Accuracy: {accuracy:.4f}")
    print("Generated files:")
    print("- training_history.png")
    print("- confusion_matrix.png")
    print("- sample_predictions.png") 
    print("- dog_cat_model.h5")

if __name__ == "__main__":
    main() 