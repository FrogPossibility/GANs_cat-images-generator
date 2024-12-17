#v5

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from PIL import Image
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale=1./255)
data_generator = data_gen.flow_from_directory('path_to_images', target_size=(64, 64), batch_size=batch_size)

# Parametri
IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10000
LR_D = 0.0002  # Learning rate per il discriminatore
LR_G = 0.0002  # Learning rate per il generatore

# Creazione del generatore
def build_generator():
    model = Sequential()
    model.add(Dense(128 * 16 * 16, activation="relu", input_dim=LATENT_DIM))
    model.add(Reshape((16, 16, 128)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu"))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same", activation="tanh"))
    model.compile(loss="binary_crossentropy", optimizer=Adam(LR_G))
    return model

# Creazione del discriminatore
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=Adam(LR_D), metrics=['accuracy'])
    return model

# Creazione del GAN
def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    model.compile(loss="binary_crossentropy", optimizer=Adam(LR_G))
    return model

# Caricamento delle immagini
def load_images(path, img_size):
    images = []
    for filename in os.listdir(path):
        try:
            img = Image.open(os.path.join(path, filename)).resize((img_size, img_size))
            img = np.array(img)
            if img.shape == (img_size, img_size, 3):  # Controllo dimensioni corrette
                images.append(img)
        except Exception as e:
            print(f"Impossibile aprire il file {filename}: {e}")
    images = np.array(images)
    images = (images - 127.5) / 127.5  # Normalizzazione tra -1 e 1
    return images

# Salvataggio delle immagini generate
def save_generated_images(epoch, generator, latent_dim):
    noise = np.random.normal(0, 1, (25, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale to [0, 1]

    for i in range(generated_images.shape[0]):
        img = (generated_images[i] * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"generated_cat_epoch_{epoch}.png")

def save_models(generator, discriminator, gan, epoch):
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    generator.save(os.path.join(save_dir, f'generator_epoch_{epoch}.h5'))
    discriminator.save(os.path.join(save_dir, f'discriminator_epoch_{epoch}.h5'))
    gan.save(os.path.join(save_dir, f'gan_epoch_{epoch}.h5'))
    
    print(f"Models saved at epoch {epoch}")

# Allenamento della GAN
def train_gan(gan, generator, discriminator, data_generator, latent_dim, epochs, batch_size):
    for epoch in range(epochs):
        # Get a batch of real images
        real_images, _ = next(data_generator)  # Keras generator already returns batches
        real_labels = np.ones((batch_size, 1))  # Labels for real images
        
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))  # Labels for fake images
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
        # Update the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gan_labels = np.ones((batch_size, 1))  # Pretend fake images are real
        g_loss = gan.train_on_batch(noise, gan_labels)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
            save_generated_images(epoch, generator, latent_dim)
        
        if epoch % 500 == 0: #ogni 100 epoche salva il modello attuale
            save_models(generator, discriminator, gan, epoch)

def load_saved_models(epoch):
    save_dir = 'saved_models'
    
    generator = load_model(os.path.join(save_dir, f'generator_epoch_{epoch}.h5'))
    discriminator = load_model(os.path.join(save_dir, f'discriminator_epoch_{epoch}.h5'))
    gan = load_model(os.path.join(save_dir, f'gan_epoch_{epoch}.h5'))
    
    print(f"Models loaded from epoch {epoch}")
    return generator, discriminator, gan

def generate_images(generator, latent_dim, num_images=5):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Riscalare tra 0 e 1
    for i in range(num_images):
        img = Image.fromarray((generated_images[i] * 255).astype(np.uint8))
        img.save(f"generated_cat_{i}.png")

# NUOVO MODELLO
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)
dataset = load_images('C:/Users/Simone/OneDrive/Desktop/GAN_IMG_GEN_CATS/cats', IMG_SIZE)
train_gan(gan, generator, discriminator, dataset, LATENT_DIM, EPOCHS, BATCH_SIZE)

# CARICARE UN MODELLO - GENERARE IMMAGINI
#epoch_to_load = 1000  # Sostituisci con l'epoca da cui vuoi caricare il modello
#generator, _, _ = load_saved_models(epoch_to_load)
#generate_images(generator, LATENT_DIM, num_images=10)

# CARICARE UN MODELLO - CONTINUARE L'ADDESTRAMENTO
#epoch_to_load = 1000  # Sostituisci con l'epoca da cui vuoi caricare il modello
#generator, discriminator, gan = load_saved_models(epoch_to_load)
#dataset = load_images('C:/Users/Simone/OneDrive/Desktop/GAN_IMG_GEN_CATS/cats', IMG_SIZE)
#remaining_epochs = EPOCHS - epoch_to_load
#train_gan(gan, generator, discriminator, dataset, LATENT_DIM, remaining_epochs, BATCH_SIZE)

# Decommentare l'esempio che si vuole utilizzare e commentare l'altro
