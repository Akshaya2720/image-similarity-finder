import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import filedialog
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity


# Load pretrained MobileNetV2
model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

IMG_SIZE = (224, 224)


# Convert an image into a 1280-dimensional embedding
def get_embedding(image_path):

    img = image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    embedding = model.predict(
        img_array,
        verbose=0
    )

    return embedding.flatten()


# Find all dataset images
image_paths = []

for root, dirs, files in os.walk("dataset"):

    for file in files:

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_paths.append(
                os.path.normpath(
                    os.path.join(root, file)
                )
            )


print("Total images:", len(image_paths))


# Generate embeddings for all dataset images
embeddings = []

for path in image_paths:

    embeddings.append(
        get_embedding(path)
    )


embeddings = np.array(embeddings)

print("Embeddings shape:", embeddings.shape)


# Open file picker
root = tk.Tk()
root.withdraw()

query_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png"),
        ("All files", "*.*")
    ]
)

root.destroy()


# Check whether an image was selected
if not query_path:

    print("No image selected.")
    exit()


query_path = os.path.normpath(query_path)


# Generate embedding for selected image
query_embedding = get_embedding(
    query_path
)


# Compare query image with dataset
similarities = cosine_similarity(
    [query_embedding],
    embeddings
)[0]


# Sort from highest similarity to lowest
top_indices = np.argsort(
    similarities
)[::-1]


# Print top 5 results
print("\nTop 5 similar images:")

for index in top_indices[:5]:

    print(
        image_paths[index],
        "->",
        round(
            similarities[index] * 100,
            2
        ),
        "%"
    )


# Display results
plt.figure(figsize=(15, 5))


# Display selected query image
plt.subplot(1, 6, 1)

plt.imshow(
    image.load_img(query_path)
)

plt.title("Query Image")

plt.axis("off")


# Display top 5 similar images
for position, index in enumerate(
    top_indices[:5],
    start=2
):

    plt.subplot(
        1,
        6,
        position
    )

    plt.imshow(
        image.load_img(
            image_paths[index]
        )
    )

    plt.title(
        f"{similarities[index] * 100:.1f}%"
    )

    plt.axis("off")


plt.tight_layout()

plt.show()