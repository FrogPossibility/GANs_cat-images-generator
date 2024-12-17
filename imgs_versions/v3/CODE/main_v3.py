import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageTk
import os
import tkinter as tk
import threading

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
def save_generated_image(epoch, generator, latent_dim):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise)[0]
    generated_image = (generated_image + 1) / 2.0  # Rescale to [0, 1]
    img = (generated_image * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f"generated_cat_epoch_{epoch}.png")

# Aggiorna l'immagine nel canvas di Tkinter
def update_image(canvas, generator, latent_dim):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise)[0]
    generated_image = (generated_image + 1) / 2.0  # Rescale to [0, 1]
    generated_image = (generated_image * 255).astype(np.uint8)
    generated_image = Image.fromarray(generated_image)
    generated_image = ImageTk.PhotoImage(generated_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=generated_image)
    canvas.image = generated_image  # Per evitare che l'immagine venga garbage collected

# Allenamento della GAN con Tkinter e Multithreading
def train_gan(gan, generator, discriminator, dataset, latent_dim, epochs, batch_size, canvas):
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

        # Aggiorna l'immagine in Tkinter
        if epoch % 1 == 0:  # Aggiorna ogni 10 epoche
            update_image(canvas, generator, latent_dim)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
            save_generated_image(epoch, generator, latent_dim)  # Salva l'immagine generata

# Funzione per avviare l'allenamento in un thread separato
def start_training_thread():
    training_thread = threading.Thread(
        target=train_gan, 
        args=(gan, generator, discriminator, dataset, LATENT_DIM, EPOCHS, BATCH_SIZE, canvas)
    )
    training_thread.start()

# Inizializzazione di Tkinter
root = tk.Tk()
root.title("GAN Image Generator")
canvas = tk.Canvas(root, width=IMG_SIZE, height=IMG_SIZE)
canvas.pack()

# Creazione dei modelli
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Caricamento del dataset
dataset = load_images('C:/Users/Simone/OneDrive/Desktop/GAN_IMG_GEN_CATS/cats', IMG_SIZE)

# Avvia il training in un thread separato
start_training_thread()

# Avvia il loop principale di Tkinter
root.mainloop()
