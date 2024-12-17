import customtkinter
import torch
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision.utils import save_image
import torchvision.transforms as transforms  # Assicurati che sia importato
import matplotlib.pyplot as plt

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("700x500")

# Inizializza le variabili
model_loaded = False
model_path = None
LATENT_DIM = 32
IMG_SIZE = 64
IMG_CHANNELS = 3
device = torch.device('cpu')  # Forziamo l'uso della CPU per questo esempio

# Creazione del generatore
class EfficientGenerator(torch.nn.Module):
    def __init__(self, latent_dim):
        super(EfficientGenerator, self).__init__()
        self.init_size = IMG_SIZE // 16
        self.l1 = torch.nn.Sequential(torch.nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, 32, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32, 0.8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(32, IMG_CHANNELS, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Inizializza il generatore
generator = EfficientGenerator(LATENT_DIM)

# Funzione per caricare il modello
def load_model():
    global model_loaded, model_path, generator
    model_path = filedialog.askopenfilename(
        title="Seleziona il modello GAN (.pth)",
        filetypes=(("Modelli PyTorch", "*.pth"), ("Tutti i file", "*.*"))
    )
    
    if model_path:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            model_loaded = True
            print(f"Modello caricato da {model_path}")
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            model_loaded = False

# Funzione per generare immagini
def generate_image(epoch):
    global model_loaded, model_path
    if not model_loaded:
        return
    
    # Genera l'immagine dal rumore z
    z = torch.randn(1, LATENT_DIM, device='cpu')
    gen_img = generator(z).detach().cpu()
    
    # Scala i valori da [-1, 1] a [0, 1] per la visualizzazione corretta
    gen_img = (gen_img + 1) / 2
    
    # Converte in formato PIL
    img_pil = transforms.ToPILImage()(gen_img.squeeze())
    img_pil_resized = img_pil.resize((256, 256))
    
    # Mostra l'immagine nella finestra tkinter
    img_tk = ImageTk.PhotoImage(img_pil_resized)
    label_image.configure(image=img_tk)
    label_image.image = img_tk

# Layout CustomTkinter
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

# Pulsante per caricare il modello
load_button = customtkinter.CTkButton(master=frame, text="Carica Modello", command=load_model)
load_button.pack(pady=10)

# Etichetta per visualizzare l'immagine generata
img_label = customtkinter.CTkLabel(master=frame)
img_label.pack(pady=10)

# Pulsante per generare l'immagine
generate_button = customtkinter.CTkButton(master=frame, text="Genera Immagine", command=generate_image)
generate_button.pack(pady=10)

root.mainloop()

# Aggiungi questa linea subito dopo root.mainloop() o nel tuo layout
label_image = customtkinter.CTkLabel(master=root)
label_image.pack(pady=20)  # Puoi regolare il padding secondo necessità