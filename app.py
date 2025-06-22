
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from typing import List

MODEL_PATH = "mnist_cgan_generator.pth"  # make sure this file is in the same repo
IMG_SIZE = 28
NZ = 100  # latent vector size


class Generator(nn.Module):
    """Conditional GAN generator matching the uploaded weights."""

    def __init__(self, nz: int = NZ, n_classes: int = 10, img_size: int = IMG_SIZE):
        super().__init__()
        self.nz = nz
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(nz + n_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, img_size * img_size),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # One‑hot like embedding concat
        c = self.label_emb(labels)
        x = torch.cat((z, c), dim=1)
        out = self.model(x)
        out = out.view(-1, 1, IMG_SIZE, IMG_SIZE)
        return out


@st.cache_resource(show_spinner="Loading generator model…")
def load_generator() -> Generator:
    """Load the pre‑trained generator model once and cache it."""
    model = Generator()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_images(gen: Generator, digit: int, n: int = 5) -> List[Image.Image]:
    """Generate *n* PIL images of the requested digit."""
    z = torch.randn(n, NZ)
    labels = torch.full((n,), digit, dtype=torch.long)
    with torch.no_grad():
        imgs = gen(z, labels).cpu()
    imgs = (imgs + 1) / 2  # [-1,1] → [0,1]
    pil = [T.ToPILImage()(img.squeeze(0)) for img in imgs]
    return pil


# ───────────────────────────── Streamlit UI ──────────────────────────────

st.set_page_config(page_title="MNIST Digit Generator", page_icon="🔢", layout="wide")
st.title("🔢 MNIST‑style Hand‑written Digit Generator")

st.markdown(
    "Pick a digit and click **Generate** to create five synthetic hand‑written images "
    "produced by a lightweight Conditional GAN trained on the MNIST dataset."
)

# Preserve last choice across reruns so successive users see previous digit
if "selected_digit" not in st.session_state:
    st.session_state.selected_digit = 0

chosen_digit = st.selectbox("Choose digit (0–9)", list(range(10)), index=st.session_state.selected_digit, key="digit_select")

if st.button("Generate"):
    st.session_state.selected_digit = chosen_digit

    gen = load_generator()
    images = generate_images(gen, chosen_digit, n=5)

    st.subheader(f"Generated images for digit **{chosen_digit}**")
    cols = st.columns(5)
    for i, img in enumerate(images):
        cols[i].image(img, width=150, caption=f"{chosen_digit}-{i+1}")

st.markdown("---")
st.markdown(
    "*Model:* Conditional GAN with a 100‑dim latent vector, trained for 30 epochs on "
    "the MNIST training set.\n\n"
    "*Availability:* This app will remain publicly accessible for **at least two weeks** "
    "from deployment. If it goes to sleep due to inactivity, simply reload; cold‑start "
    "time is typically under 30 seconds on the Streamlit Community Cloud free tier."
)
