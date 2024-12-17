import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from PIL import Image
import os

# Parametri
IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 64 #prova 128.
EPOCHS = 10000
LR_D = 0.0002  # Learning rate per il discriminatore
LR_G = 0.0002  # Learning rate per il generatore

# Creazione del generatore
def build_generator():
    model = Sequential()
    model.add(Dense(256 * 16 * 16, activation="relu", input_dim=LATENT_DIM))
    model.add(Reshape((16, 16, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"))  # Output: 32x32x128
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"))   # Output: 64x64x64
    model.add(Conv2D(3, kernel_size=3, padding="same", activation="tanh"))  # Output: 64x64x3
    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(64, 64, 3)))  # Input: 64x64x3, Output: 32x32x64
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))  # Output: 16x16x128
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))  # Output: 8x8x256
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())  # Output: 8*8*256 = 16384
    model.add(Dense(1, activation="sigmoid"))
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

# Allenamento della GAN
def train_gan(gan, generator, discriminator, dataset, latent_dim, epochs, batch_size):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Aggiorna il discriminatore
        for _ in range(2):  # Aggiorna il discriminatore due volte per ogni aggiornamento del generatore
            idx = np.random.randint(0, dataset.shape[0], half_batch)
            real_images = dataset[idx]
            noise = np.random.normal(0, 1, (half_batch, latent_dim))
            fake_images = generator.predict(noise)

            d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])

        # Aggiorna il generatore
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
            save_generated_images(epoch, generator, latent_dim)

# Creazione dei modelli

discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(LR_D), metrics=['accuracy'])

generator = build_generator()

gan = build_gan(generator, discriminator)
gan.compile(loss="binary_crossentropy", optimizer=Adam(LR_G))

# Caricamento del dataset e training
dataset = load_images('C:/Users/Simone/OneDrive/Desktop/GAN_IMG_GEN_CATS/cats', IMG_SIZE)
train_gan(gan, generator, discriminator, dataset, LATENT_DIM, EPOCHS, BATCH_SIZE)





# Genera nuove immagini di gatti
def generate_images(generator, latent_dim, num_images=5):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Riscalare tra 0 e 1
    for i in range(num_images):
        img = Image.fromarray((generated_images[i] * 255).astype(np.uint8))
        img.save(f"generated_cat_{i}.png")

# Generazione di immagini
generate_images(generator, LATENT_DIM, num_images=5)
